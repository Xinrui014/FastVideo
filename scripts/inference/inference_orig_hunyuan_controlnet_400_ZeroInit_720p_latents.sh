#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
num_gpus=1
export MODEL_BASE=/data/atlas/projects/FastVideo/data/hunyuan/
torchrun --nnodes=1 --nproc_per_node=$num_gpus --master_port 29512 \
    fastvideo/sample/sample_t2v_hunyuan_from_720p_latents.py \
    --height 720 \
    --width 1280 \
    --num_frames 45 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt /data/atlas/projects/FastVideo/assets/prompt_1-2.txt \
    --seed 1024 \
    --output_path /data/atlas/projects/FastVideo/outputs_video/hunyuan/controlnet_ZeroInit_step50_latentLR/ \
    --model_path /data/atlas/projects/FastVideo/data/hunyuan \
    --dit-weight /data/atlas/projects/FastVideo/data/outputs/Hunyuan_ControlNet_Finetune_zeroconv_blocks_6/checkpoint-400/diffusion_pytorch_model.safetensors \
    --vae-sp
