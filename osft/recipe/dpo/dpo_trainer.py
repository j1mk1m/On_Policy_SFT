import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import logging
import json
from typing import Dict, Optional, Tuple

import hydra
import torch
import torch.nn.functional as F
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None

import verl.utils.hdfs_io as hdfs_io
from verl.utils.dataset import RMDataset
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, get_init_weight_context_manager, init_fn
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_DPO_LOGGING_LEVEL", "WARN"))


def _assert_finite(tensor: torch.Tensor, name: str):
    assert torch.isfinite(tensor).all(), f"Non-finite values detected in {name}"


def _sequence_log_probs(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    labels = input_ids[:, 1:].contiguous()
    shift_logits = logits[:, :-1, :].contiguous().to(torch.float32)
    shift_loss_mask = loss_mask[:, 1:].contiguous().to(shift_logits.dtype)

    token_logps = torch.gather(shift_logits.log_softmax(dim=-1), dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    seq_logps = (token_logps * shift_loss_mask).sum(dim=-1)
    _assert_finite(seq_logps, "sequence_logps")
    return seq_logps


def dpo_preference_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    if reference_free:
        ref_logratios = 0.0

    logits = pi_logratios - ref_logratios
    _assert_finite(logits, "dpo_logits")
    z = beta * logits
    losses = -F.logsigmoid(z) * (1 - label_smoothing) - F.logsigmoid(-z) * label_smoothing
    _assert_finite(losses, "dpo_losses")
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses, chosen_rewards, rejected_rewards, z


class FSDPDPOTrainer:
    def __init__(self, config, device_mesh: DeviceMesh, tokenizer, train_dataset: Dataset, val_dataset: Dataset):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer
        self.global_step = 0
        self.resume_epoch = 0
        self.resume_step_in_epoch = 0
        self._normalize_config_bsz()
        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()
        self._warned_no_response_mask = False
        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = get_device_name()

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        self.config.data.train_batch_size //= dp_size
        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset: Dataset, val_dataset: Dataset):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()
        if rank == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True)
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True)
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _load_causal_lm(self, local_model_path: str, trust_remote_code: bool, torch_dtype: torch.dtype) -> PreTrainedModel:
        model_config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.device_mesh)
        with init_context():
            model = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=model_config,
                torch_dtype=torch_dtype,
                attn_implementation=self.config.model.get("attn_implementation", "eager"),
                trust_remote_code=trust_remote_code,
            )
        return model

    def _build_model_optimizer(self):
        resume_from = self.config.trainer.get("resume_from", None)
        model_source = resume_from if resume_from is not None else self.config.model.partial_pretrain
        local_model_path = copy_to_local(src=model_source, verbose=True)
        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = PrecisionType.to_dtype(self.config.model.fsdp_config.get("model_dtype", "fp32"))

        self.policy_model = self._load_causal_lm(local_model_path=local_model_path, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
        self.reference_model = self._load_causal_lm(local_model_path=local_model_path, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype)
        self.model_config = self.policy_model.config

        use_lora = bool(self.config.model.get("lora", {}).get("enabled", False))
        if use_lora:
            assert get_peft_model is not None, "PEFT is not installed. Install `peft` to enable LoRA training."
            if self.device_mesh.get_rank() == 0:
                print("Applying LoRA/PEFT to policy model")
            self.policy_model.enable_input_require_grads()
            lora_cfg = self.config.model.lora
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora_cfg.r),
                lora_alpha=int(lora_cfg.alpha),
                lora_dropout=float(lora_cfg.dropout),
                target_modules=list(lora_cfg.target_modules) if lora_cfg.target_modules is not None else None,
                bias=str(lora_cfg.bias),
            )
            self.policy_model = get_peft_model(self.policy_model, lora_config)
            if self.device_mesh.get_rank() == 0:
                self.policy_model.print_trainable_parameters()

        if self.config.model.enable_gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32)
        auto_wrap_policy = get_fsdp_wrap_policy(
            self.policy_model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=use_lora,
        )

        self.fsdp_policy = FSDP(
            self.policy_model,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )
        self.fsdp_reference = FSDP(
            self.reference_model,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )
        self.fsdp_reference.eval()
        for param in self.fsdp_reference.parameters():
            param.requires_grad = False

        trainable_params = [p for p in self.fsdp_policy.parameters() if p.requires_grad]
        assert len(trainable_params) > 0, "No trainable parameters found; check LoRA/PEFT config."
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            weight_decay=self.config.optim.weight_decay,
        )

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        if self.device_mesh.get_rank() == 0:
            print(f"Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}")

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)
        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps)
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

        self._maybe_load_resume_state()
        log_gpu_memory_usage("After model and optimizer initialization", logger=logger)

    def _maybe_load_resume_state(self):
        resume_from = self.config.trainer.get("resume_from", None)
        if resume_from is None:
            return

        rank = self.device_mesh.get_rank()
        trainer_state_path = os.path.join(resume_from, "trainer_state.json")
        scheduler_state_path = os.path.join(resume_from, "scheduler.pt")
        optimizer_state_path = os.path.join(resume_from, f"optimizer_rank{rank}.pt")

        if rank == 0:
            assert os.path.exists(trainer_state_path), f"Missing trainer state at {trainer_state_path}"
            assert os.path.exists(scheduler_state_path), f"Missing scheduler state at {scheduler_state_path}"
            assert os.path.exists(optimizer_state_path), f"Missing optimizer state at {optimizer_state_path}"
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
        else:
            trainer_state = None

        object_list = [trainer_state]
        torch.distributed.broadcast_object_list(object_list, src=0)
        trainer_state = object_list[0]

        self.global_step = int(trainer_state["global_step"])
        self.resume_epoch = int(trainer_state["epoch"])
        self.resume_step_in_epoch = int(trainer_state["step_in_epoch"])

        optimizer_state = torch.load(optimizer_state_path, map_location="cpu")
        self.optimizer.load_state_dict(optimizer_state)

        scheduler_state = torch.load(scheduler_state_path, map_location="cpu")
        self.lr_scheduler.load_state_dict(scheduler_state)

        if rank == 0:
            print(
                f"Resumed from {resume_from} "
                f"(global_step={self.global_step}, epoch={self.resume_epoch}, step_in_epoch={self.resume_step_in_epoch})"
            )

    def _compute_batch_loss(self, batch: Dict[str, torch.Tensor], do_backward: bool = False):
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)

        chosen_input_ids = input_ids[:, 0, :]
        rejected_input_ids = input_ids[:, 1, :]
        chosen_attention_mask = attention_mask[:, 0, :]
        rejected_attention_mask = attention_mask[:, 1, :]
        chosen_loss_mask, rejected_loss_mask = self._build_response_loss_masks(batch, attention_mask)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            policy_chosen_logits = self.fsdp_policy(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False).logits
            policy_rejected_logits = self.fsdp_policy(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False).logits

        policy_chosen_logps = _sequence_log_probs(policy_chosen_logits, chosen_input_ids, chosen_loss_mask)
        policy_rejected_logps = _sequence_log_probs(policy_rejected_logits, rejected_input_ids, rejected_loss_mask)

        with torch.no_grad(), torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            ref_chosen_logits = self.fsdp_reference(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False).logits
            ref_rejected_logits = self.fsdp_reference(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask, use_cache=False).logits
            ref_chosen_logps = _sequence_log_probs(ref_chosen_logits, chosen_input_ids, chosen_loss_mask)
            ref_rejected_logps = _sequence_log_probs(ref_rejected_logits, rejected_input_ids, rejected_loss_mask)

        losses, chosen_rewards, rejected_rewards, z = dpo_preference_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=self.config.loss.beta,
            label_smoothing=self.config.loss.label_smoothing,
            reference_free=self.config.loss.reference_free,
        )
        loss = losses.mean()
        _assert_finite(loss, "batch_loss")
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        z_mean = z.mean()
        z_std = z.std(unbiased=False)
        z_max = z.max()
        z_min = z.min()
        return loss, chosen_rewards.mean(), rejected_rewards.mean(), reward_accuracies.mean(), z_mean, z_std, z_max, z_min

    def _extract_prompt_lengths(self, batch: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        prompt_length_keys = (
            "prompt_length",
            "prompt_lengths",
            "prompt_len",
            "prompt_lens",
            "prompt_ids_lens",
            "prompt_token_lens",
        )
        for key in prompt_length_keys:
            if key in batch:
                return torch.as_tensor(batch[key], device=self.device_name)
        return None

    def _build_response_mask_from_prompt_lengths(self, prompt_lengths: torch.Tensor, pair_attention_mask: torch.Tensor) -> torch.Tensor:
        seq_len = pair_attention_mask.size(-1)
        prompt_lengths = prompt_lengths.clamp(min=0, max=seq_len).to(torch.long)
        token_positions = torch.arange(seq_len, device=self.device_name).unsqueeze(0).expand_as(pair_attention_mask)
        response_mask = token_positions >= prompt_lengths.unsqueeze(-1)
        return response_mask.to(pair_attention_mask.dtype) * pair_attention_mask.to(torch.float32)

    def _build_response_loss_masks(self, batch: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if "response_mask" in batch:
            response_mask = torch.as_tensor(batch["response_mask"], device=self.device_name)
            if response_mask.ndim == 3 and response_mask.size(1) == 2:
                return response_mask[:, 0, :], response_mask[:, 1, :]
            if response_mask.ndim == 2:
                return response_mask, response_mask
            raise ValueError(f"Unexpected response_mask shape: {tuple(response_mask.shape)}")

        if "loss_mask" in batch:
            loss_mask = torch.as_tensor(batch["loss_mask"], device=self.device_name)
            if loss_mask.ndim == 3 and loss_mask.size(1) == 2:
                return loss_mask[:, 0, :], loss_mask[:, 1, :]
            if loss_mask.ndim == 2:
                return loss_mask, loss_mask
            raise ValueError(f"Unexpected loss_mask shape: {tuple(loss_mask.shape)}")

        prompt_lengths = self._extract_prompt_lengths(batch)
        if prompt_lengths is not None:
            if prompt_lengths.ndim == 2 and prompt_lengths.size(1) == 2:
                chosen_prompt_lengths = prompt_lengths[:, 0]
                rejected_prompt_lengths = prompt_lengths[:, 1]
            elif prompt_lengths.ndim == 1:
                chosen_prompt_lengths = prompt_lengths
                rejected_prompt_lengths = prompt_lengths
            else:
                raise ValueError(f"Unexpected prompt length tensor shape: {tuple(prompt_lengths.shape)}")

            chosen_mask = self._build_response_mask_from_prompt_lengths(chosen_prompt_lengths, attention_mask[:, 0, :])
            rejected_mask = self._build_response_mask_from_prompt_lengths(rejected_prompt_lengths, attention_mask[:, 1, :])
            return chosen_mask, rejected_mask

        if not self._warned_no_response_mask and self.device_mesh.get_rank() == 0:
            logger.warning("No response/loss mask or prompt lengths found in batch; falling back to attention_mask (includes prompt tokens).")
            self._warned_no_response_mask = True
        return attention_mask[:, 0, :], attention_mask[:, 1, :]

    def training_step(self, batch: Dict[str, torch.Tensor]):
        self.fsdp_policy.train()
        self.optimizer.zero_grad()
        n_micro_batches = self.config.data.train_batch_size // self.config.data.micro_batch_size_per_gpu
        step_loss = 0.0
        step_chosen_reward = 0.0
        step_rejected_reward = 0.0
        step_reward_acc = 0.0
        step_z_mean = 0.0
        step_z_std = 0.0
        step_z_max = 0.0
        step_z_min = 0.0

        for i in range(n_micro_batches):
            start = i * self.config.data.micro_batch_size_per_gpu
            end = (i + 1) * self.config.data.micro_batch_size_per_gpu
            micro_batch = {k: v[start:end] for k, v in batch.items()}
            loss, chosen_reward, rejected_reward, reward_acc, z_mean, z_std, z_max, z_min = self._compute_batch_loss(micro_batch, do_backward=False)
            (loss / n_micro_batches).backward()
            step_loss += loss.item()
            step_chosen_reward += chosen_reward.item()
            step_rejected_reward += rejected_reward.item()
            step_reward_acc += reward_acc.item()
            step_z_mean += z_mean.item()
            step_z_std += z_std.item()
            step_z_max += z_max.item()
            step_z_min += z_min.item()

        grad_norm = self.fsdp_policy.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        self.optimizer.step()
        self.lr_scheduler.step()

        metrics_tensor = torch.tensor(
            [
                step_loss / n_micro_batches,
                step_chosen_reward / n_micro_batches,
                step_rejected_reward / n_micro_batches,
                step_reward_acc / n_micro_batches,
                step_z_mean / n_micro_batches,
                step_z_std / n_micro_batches,
                step_z_max / n_micro_batches,
                step_z_min / n_micro_batches,
            ],
            device=self.device_name,
        )
        if is_cuda_available:
            torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(metrics_tensor)
            metrics_tensor /= self.device_mesh.size(0)

        lr = self.lr_scheduler.get_last_lr()[0]
        return {
            "train/loss": metrics_tensor[0].item(),
            "train/reward_chosen": metrics_tensor[1].item(),
            "train/reward_rejected": metrics_tensor[2].item(),
            "train/reward_accuracy": metrics_tensor[3].item(),
            "train/z_mean": metrics_tensor[4].item(),
            "train/z_std": metrics_tensor[5].item(),
            "train/z_max": metrics_tensor[6].item(),
            "train/z_min": metrics_tensor[7].item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    def validation_step(self, batch: Dict[str, torch.Tensor]):
        self.fsdp_policy.eval()
        with torch.no_grad():
            loss, chosen_reward, rejected_reward, reward_acc, z_mean, z_std, z_max, z_min = self._compute_batch_loss(batch, do_backward=False)
            metrics_tensor = torch.tensor(
                [
                    loss.item(),
                    chosen_reward.item(),
                    rejected_reward.item(),
                    reward_acc.item(),
                    z_mean.item(),
                    z_std.item(),
                    z_max.item(),
                    z_min.item(),
                ],
                device=self.device_name,
            )
            if is_cuda_available:
                torch.distributed.all_reduce(metrics_tensor, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(metrics_tensor)
                metrics_tensor /= self.device_mesh.size(0)
        return metrics_tensor

    def save_checkpoint(self, step: int, epoch: int, step_in_epoch: int):
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        # Every rank writes rank-local optimizer state into this directory.
        os.makedirs(path, exist_ok=True)
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_policy, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_policy.state_dict()
        if self.device_mesh.get_rank() == 0:
            self.policy_model.save_pretrained(path, state_dict=state_dict)
            self.model_config.save_pretrained(path)
            self.tokenizer.save_pretrained(path)

        optimizer_state = self.optimizer.state_dict()
        optimizer_output = os.path.join(path, f"optimizer_rank{self.device_mesh.get_rank()}.pt")
        torch.save(optimizer_state, optimizer_output)

        if self.device_mesh.get_rank() == 0:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(path, "scheduler.pt"))
            trainer_state = {
                "global_step": int(step),
                "epoch": int(epoch),
                "step_in_epoch": int(step_in_epoch),
            }
            with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as f:
                json.dump(trainer_state, f)

        # Ensure every rank finished writing rank-local optimizer state files.
        torch.distributed.barrier()
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        if rank == 0 and self.config.trainer.get("resume_from", None) is not None:
            print(
                f"Continue training from global_step={self.global_step}, "
                f"resume_epoch={self.resume_epoch}, resume_step_in_epoch={self.resume_step_in_epoch}"
            )

        for epoch in range(self.resume_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            start_step_in_epoch = self.resume_step_in_epoch + 1 if epoch == self.resume_epoch else 1
            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                ),
                start=1,
            ):
                if step_in_epoch < start_step_in_epoch:
                    continue

                self.global_step += 1
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=self.global_step)

                is_last_step = self.global_step >= self.total_training_steps
                is_valid_step = self.config.trainer.test_freq > 0 and self.global_step % self.config.trainer.test_freq == 0
                is_save_step = self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0

                if is_last_step or is_valid_step:
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        val_metrics.append(self.validation_step(val_data))
                    if rank == 0:
                        stacked = torch.stack(val_metrics).mean(dim=0)
                        tracking.log(
                            data={
                                "val/loss": stacked[0].item(),
                                "val/reward_chosen": stacked[1].item(),
                                "val/reward_rejected": stacked[2].item(),
                                "val/reward_accuracy": stacked[3].item(),
                                "val/z_mean": stacked[4].item(),
                                "val/z_std": stacked[5].item(),
                                "val/z_max": stacked[6].item(),
                                "val/z_min": stacked[7].item(),
                            },
                            step=self.global_step,
                        )
                    torch.distributed.barrier()

                if is_last_step or is_save_step:
                    self.save_checkpoint(step=self.global_step, epoch=epoch, step_in_epoch=step_in_epoch)

                if is_last_step:
                    return

            # Resume skip is only relevant for the first resumed epoch.
            self.resume_step_in_epoch = 0


def create_dpo_dataset(data_paths, data_config, tokenizer):
    return RMDataset(
        parquet_files=data_paths,
        tokenizer=tokenizer,
        prompt_key=data_config.prompt_key,
        chosen_key=data_config.chosen_key,
        rejected_key=data_config.rejected_key,
        max_length=data_config.max_length,
        add_eos=data_config.add_eos,
        cache_dir=data_config.cache_dir,
    )


def run_dpo(config):
    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))

    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_dpo_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_dpo_dataset(config.data.val_files, config.data, tokenizer)

    trainer = FSDPDPOTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.fit()
    destroy_global_process_group()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


if __name__ == "__main__":
    main()
