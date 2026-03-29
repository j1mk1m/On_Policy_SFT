set -e
set -x
VISIBLE_DEVICES="0,1,2,3"
export HYDRA_FULL_ERROR=1
# export WORLD_SIZE=1
# export RANK=0
# export LOCAL_RANK=0


ROOT_DIR=$(pwd)
DATA_DIR=/data/user_data/gyeongwk
STRING_TASK_PATH=$HOME/RL-Compositionality/data/string_task
TRAIN_FILE=$STRING_TASK_PATH/stage2_level2/train.parquet
VAL_FILE=$STRING_TASK_PATH/stage2_level1to8/test.parquet

# VAL_PREFIX=$ROOT_DIR/data/benchmarks
# MATH500_PATH=$VAL_PREFIX/math500.parquet
# MATH500_100_PATH=$VAL_PREFIX/math500_100.parquet
# AIME_PATH=$VAL_PREFIX/aime.parquet
# AIME25_PATH=$VAL_PREFIX/aime25.parquet
# AMC_PATH=$VAL_PREFIX/amc.parquet
# OLYMPIAD_PATH=$VAL_PREFIX/olympiadbench.parquet
# MINERVA_PATH=$VAL_PREFIX/minerva.parquet

# VAL_FILE_LIST="['$MATH500_PATH','$AIME25_PATH']"


LR=1e-6
BACKBONE="stage1-rft"
BACKBONE_PATH=$DATA_DIR/checkpoints/string-task/stage1-rft
MAX_PROMPT_LENGTH=1024
MAX_GEN_LENGTH=2048
MODEL_ID="llama-3.1-8b-stage1-rft"
DATE=$(date +"%m%d_%H%M")
TASK="OSFT"
DATASET_NAME="string-task"
ROLLOUT_N=4
EXPERIMENT="OSFT-${DATASET_NAME}"
ENABLE_TRAIN_TEMP=False
TAU_S=1

PROJECT_NAME="string-task"

EXP="${TASK}-${MODEL_ID}-${EXPERIMENT}-lr${LR}-TAUS${TAU_S}-rollout${ROLLOUT_N}-ttr${ENABLE_TRAIN_TEMP}-${DATE}"
OUTPUT_DIR="${DATA_DIR}/checkpoints/${PROJECT_NAME}/${EXP}"


CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} \
python3 -m recipe.osft.main_osft \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=4 \
    data.filter_overlong_prompts=True \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_GEN_LENGTH} \
    actor_rollout_ref.model.path=${BACKBONE_PATH} \
    actor_rollout_ref.model.use_liger=False \
    actor_rollout_ref.model.use_shm=False \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.temperature=${TAU_S} \
    actor_rollout_ref.rollout.val_kwargs.temperature=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    trainer.enable_train_temperature=${ENABLE_TRAIN_TEMP} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXP} \
    trainer.val_before_train=True \
    trainer.default_local_dir=${OUTPUT_DIR} \
    trainer.n_gpus_per_node=4 \
    trainer.default_hdfs_dir=null \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.rollout_data_dir=${OUTPUT_DIR}/rollout_data \
    trainer.validation_data_dir=${OUTPUT_DIR}/rollout_eval_data \
    trainer.test_freq=10 \
    +trainer.log_freq=1 \
    trainer.total_epochs=2
