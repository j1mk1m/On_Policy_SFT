set -e
set -x

export HYDRA_FULL_ERROR=1

NGPUS=4
VISIBLE_DEVICES="0,1,2,3"

USER_DATA_DIR=/data/user_data/gyeongwk

MODEL_PATH="gyeongwk/stage1-rft"
TRAIN_FILE="data/string_task/dpo-all-max-none/train.parquet"
VAL_FILE="data/string_task/dpo-all-max-none/test.parquet"
EXP_NAME="dpo-string-task-full"
OUTPUT_DIR="${USER_DATA_DIR}/checkpoints/string-task/${EXP_NAME}"

# Example: "${OUTPUT_DIR}/global_step_400"
RESUME_FROM=""
RESUME_ARGS=()
if [ -n "${RESUME_FROM}" ]; then
  RESUME_ARGS+=("trainer.resume_from=${RESUME_FROM}")
fi

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} torchrun --standalone --nproc_per_node=${NGPUS} \
  -m recipe.dpo.dpo_trainer \
  model.partial_pretrain="${MODEL_PATH}" \
  model.attn_implementation=flash_attention_2 \
  model.fsdp_config.model_dtype=bf16 \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=1 \
  data.max_length=4096 \
  loss.beta=0.01 \
  loss.label_smoothing=0.0 \
  trainer.project_name=string-task \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.default_hdfs_dir=null \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=${NGPUS} \
  trainer.save_freq=100 \
  trainer.test_freq=25 \
  trainer.total_epochs=1 \
  "${RESUME_ARGS[@]}"
