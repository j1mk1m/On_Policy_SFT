import os
import inspect
from typing import Iterable, Optional

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import torch
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import AutoTokenizer, set_seed
from trl import DPOConfig, DPOTrainer

try:
    from peft import LoraConfig
except ImportError:  # pragma: no cover - optional dependency at runtime.
    LoraConfig = None


def _as_list(paths) -> list[str]:
    if paths is None:
        return []
    if isinstance(paths, (list, tuple)):
        return [os.path.expanduser(str(path)) for path in paths]
    return [os.path.expanduser(str(paths))]


def _load_parquet_dataset(paths) -> Optional[object]:
    parquet_files = _as_list(paths)
    if not parquet_files:
        return None
    return load_dataset("parquet", data_files=parquet_files, split="train")


def _prepare_preference_dataset(dataset, data_cfg):
    if dataset is None:
        return None
    required_keys = {
        "prompt": str(data_cfg.prompt_key),
        "chosen": str(data_cfg.chosen_key),
        "rejected": str(data_cfg.rejected_key),
    }
    for standard_key, source_key in required_keys.items():
        if source_key not in dataset.column_names:
            raise KeyError(f"Missing required column '{source_key}' for '{standard_key}'.")
        if source_key != standard_key:
            dataset = dataset.rename_column(source_key, standard_key)
    return dataset


def _resolve_dtype(model_cfg) -> torch.dtype:
    dtype_name = str(model_cfg.get("dtype", model_cfg.get("fsdp_config", {}).get("model_dtype", "bf16"))).lower()
    if dtype_name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype_name in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def _compute_grad_accum_steps(data_cfg) -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    train_batch_size = int(data_cfg.train_batch_size)
    micro_batch = int(data_cfg.micro_batch_size_per_gpu)
    denom = world_size * micro_batch
    if train_batch_size % denom != 0:
        raise ValueError(
            f"data.train_batch_size={train_batch_size} must be divisible by WORLD_SIZE({world_size}) "
            f"* data.micro_batch_size_per_gpu({micro_batch})."
        )
    return max(1, train_batch_size // denom)


def _build_report_targets(logger_cfg) -> list[str]:
    if logger_cfg is None:
        return []
    values = logger_cfg if isinstance(logger_cfg, Iterable) and not isinstance(logger_cfg, str) else [logger_cfg]
    allowed = {"wandb", "tensorboard", "mlflow", "comet_ml", "clearml"}
    return [str(v) for v in values if str(v) in allowed]


def _build_peft_config(model_cfg):
    lora_cfg = model_cfg.get("lora", {})
    if not bool(lora_cfg.get("enabled", False)):
        return None
    if LoraConfig is None:
        raise ImportError("LoRA is enabled, but `peft` is not installed in this environment.")
    return LoraConfig(
        r=int(lora_cfg.get("r", 64)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        target_modules=lora_cfg.get("target_modules", None),
        task_type="CAUSAL_LM",
    )


def _filter_supported_dpo_config_kwargs(config_kwargs: dict) -> dict:
    try:
        if hasattr(DPOConfig, "__dataclass_fields__"):
            supported = set(DPOConfig.__dataclass_fields__.keys())
        else:
            supported = set(inspect.signature(DPOConfig.__init__).parameters.keys()) - {"self"}
    except Exception:
        return config_kwargs
    return {k: v for k, v in config_kwargs.items() if k in supported}


def _build_dpo_trainer_with_compatible_tokenizer_kwarg(trainer_kwargs: dict):
    try:
        signature = inspect.signature(DPOTrainer.__init__)
        if "processing_class" in signature.parameters and "tokenizer" in trainer_kwargs:
            trainer_kwargs["processing_class"] = trainer_kwargs.pop("tokenizer")
    except Exception:
        pass
    return DPOTrainer(**trainer_kwargs)


def run_dpo(cfg):
    set_seed(int(cfg.trainer.get("seed", 1)))
    train_dataset = _prepare_preference_dataset(_load_parquet_dataset(cfg.data.train_files), cfg.data)
    eval_dataset = _prepare_preference_dataset(_load_parquet_dataset(cfg.data.get("val_files", None)), cfg.data)
    gradient_accumulation_steps = _compute_grad_accum_steps(cfg.data)
    report_to = _build_report_targets(cfg.trainer.get("logger", []))

    model_name_or_path = str(cfg.model.partial_pretrain)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=bool(cfg.model.get("trust_remote_code", False)),
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = _resolve_dtype(cfg.model)
    model_init_kwargs = {
        "trust_remote_code": bool(cfg.model.get("trust_remote_code", False)),
        "torch_dtype": torch_dtype,
    }
    if cfg.model.get("attn_implementation", None):
        model_init_kwargs["attn_implementation"] = str(cfg.model.attn_implementation)

    has_eval = eval_dataset is not None and int(cfg.trainer.get("test_freq", 0)) > 0
    config_kwargs = {
        "output_dir": str(cfg.trainer.default_local_dir),
        "run_name": str(cfg.trainer.experiment_name),
        "per_device_train_batch_size": int(cfg.data.micro_batch_size_per_gpu),
        "per_device_eval_batch_size": int(cfg.data.micro_batch_size_per_gpu),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "num_train_epochs": float(cfg.trainer.total_epochs),
        "max_steps": int(cfg.trainer.total_training_steps) if cfg.trainer.total_training_steps is not None else -1,
        "learning_rate": float(cfg.optim.lr),
        "weight_decay": float(cfg.optim.weight_decay),
        "adam_beta1": float(cfg.optim.betas[0]),
        "adam_beta2": float(cfg.optim.betas[1]),
        "warmup_ratio": float(cfg.optim.warmup_steps_ratio),
        "lr_scheduler_type": str(cfg.optim.get("lr_scheduler", "cosine")),
        "max_grad_norm": float(cfg.optim.clip_grad),
        "logging_steps": int(cfg.trainer.get("logging_steps", 10)),
        "evaluation_strategy": "steps" if has_eval else "no",
        "eval_steps": int(cfg.trainer.get("test_freq", 25)),
        "save_strategy": "steps" if int(cfg.trainer.get("save_freq", 0)) > 0 else "no",
        "save_steps": int(cfg.trainer.get("save_freq", 100)),
        "gradient_checkpointing": bool(cfg.model.get("enable_gradient_checkpointing", True)),
        "bf16": torch_dtype == torch.bfloat16,
        "fp16": torch_dtype == torch.float16,
        "remove_unused_columns": False,
        "report_to": report_to,
        "ddp_find_unused_parameters": False,
        "beta": float(cfg.loss.get("beta", 0.1)),
        "label_smoothing": float(cfg.loss.get("label_smoothing", 0.0)),
        "loss_type": str(cfg.loss.get("loss_type", "sigmoid")),
        "reference_free": bool(cfg.loss.get("reference_free", False)),
        "max_length": int(cfg.data.max_length) if cfg.data.get("max_length", None) is not None else None,
        "max_prompt_length": int(cfg.data.max_prompt_length) if cfg.data.get("max_prompt_length", None) is not None else None,
        "model_init_kwargs": model_init_kwargs,
    }
    training_args = DPOConfig(**_filter_supported_dpo_config_kwargs(config_kwargs))

    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        print(OmegaConf.to_yaml(cfg))
        print(f"Using gradient_accumulation_steps={gradient_accumulation_steps} with WORLD_SIZE={os.environ.get('WORLD_SIZE', '1')}")

    trainer = _build_dpo_trainer_with_compatible_tokenizer_kwarg(
        {
            "model": model_name_or_path,
            "ref_model": None,
            "args": training_args,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "tokenizer": tokenizer,
            "peft_config": _build_peft_config(cfg.model),
        }
    )
    resume_from = cfg.trainer.get("resume_from", None)
    trainer.train(resume_from_checkpoint=resume_from if resume_from else None)
    trainer.save_model()
    trainer.save_state()


@hydra.main(config_path="config", config_name="dpo_trainer", version_base=None)
def main(config):
    run_dpo(config)


if __name__ == "__main__":
    main()
