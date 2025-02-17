#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
num_gpus=2
export MODEL_BASE=/data/atlas/projects/FastVideo/data/FastHunyuan
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29505 \
    fastvideo/sample/sample_t2v_hunyuan.py \
    --height 720 \
    --width 1280 \
    --num_frames 61 \
    --num_inference_steps 6 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 17 \
    --flow-reverse \
    --prompt ./assets/prompt.txt \
    --seed 1024 \
    --output_path outputs_video/hunyuan/vae_sp_cfg_6_1088_tf/ \
    --model_path $MODEL_BASE \
    --dit-weight ${MODEL_BASE}/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --vae-sp
