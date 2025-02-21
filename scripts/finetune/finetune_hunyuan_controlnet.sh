export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="a5ebf533c17c677bcee36f66c91907b5fb102f7c"

export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nnodes 1 --nproc_per_node 4 --master_port 29511 \
    fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path /data/atlas/projects/FastVideo/data/hunyuan \
    --dit_model_name_or_path /data/atlas/projects/FastVideo/data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan_controlnet" \
    --cache_dir data/.cache \
    --data_json_path data/Inte4K/videos2caption.json \
    --validation_prompt_dir data/Inte4K/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 12 \
    --sp_size 4 \
    --train_sp_batch_size 1 \
    --fsdp_sharding_startegy "full" \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=100 \
    --validation_steps 2000 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 10 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=data/outputs/Hunyuan_ControlNet_Finetune_zeroconv_blocks_10_gpus4 \
    --tracker_project_name Hunyuan_ControlNet_Finetune_2 \
    --num_frames 45 \
    --num_height 1088 \
    --num_width 1920 \
    --shift 7 \
    --validation_guidance_scale "1.0"