# export WANDB_MODE="offline"
export CUDA_VISIBLE_DEVICES=3
GPU_NUM=1 # 2,4,8
MODEL_PATH="data/hunyuan"
MODEL_TYPE="hunyuan"
DATA_MERGE_PATH="data/Image-Vid-Finetune-Src/merge.txt"
OUTPUT_DIR="data/Image-Vid-Finetune-HunYuan_fps_8_45_480"
VALIDATION_PATH="assets/prompt.txt"


torchrun --nproc_per_node=$GPU_NUM --master_port 29507 \
    fastvideo/data_preprocess/preprocess_vae_latents.py \
    --model_path $MODEL_PATH \
    --data_merge_path $DATA_MERGE_PATH \
    --train_batch_size=1 \
    --max_height=480 \
    --max_width=848 \
    --num_frames=45 \
    --dataloader_num_workers 1 \
    --output_dir=$OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --train_fps 8

torchrun --nproc_per_node=$GPU_NUM \
    fastvideo/data_preprocess/preprocess_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR

torchrun --nproc_per_node=1 \
    fastvideo/data_preprocess/preprocess_validation_text_embeddings.py \
    --model_type $MODEL_TYPE \
    --model_path $MODEL_PATH \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt_txt $VALIDATION_PATH