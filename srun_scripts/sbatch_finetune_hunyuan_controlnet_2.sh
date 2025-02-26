#!/bin/bash
#SBATCH --job-name=lanqing001
#SBATCH -A CGAI24022
#SBATCH --nodes=2               # Request 8 nodes
#SBATCH --ntasks-per-node=1     # Ensure 1 task per node
#SBATCH --time=1:00:00         # Max runtime
#SBATCH --partition=gh        # GPU partition (or adjust based on your system)
#SBATCH --error=/scratch/10320/lanqing001/xinrui/FastVideo/srun_scripts/output/job.%J.err
#SBATCH --output=/scratch/10320/lanqing001/xinrui/FastVideo/srun_scripts/output/job.%J.out

# Load required modules
module unload python3
module load gcc python3_mpi cuda/12.4

# Activate your Python environment
source /scratch/10320/lanqing001/python-envs/sdxl-python/bin/activate

echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"


# Run the script with `srun'
cd /scratch/10320/lanqing001/xinrui/FastVideo
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12355
export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_MODE=online
export WANDB_API_KEY="a5ebf533c17c677bcee36f66c91907b5fb102f7c"


srun -N 2 -n 2 \
    python /scratch/10320/lanqing001/xinrui/FastVideo/fastvideo/train.py \
    --seed 42 \
    --pretrained_model_name_or_path /scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan \
    --dit_model_name_or_path /scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt\
    --model_type "hunyuan_controlnet" \
    --cache_dir /scratch/10320/lanqing001/xinrui/FastVideo/data/.cache \
    --data_json_path /scratch/10320/lanqing001/xinrui/FastVideo/data/Inte4K/videos2caption.json \
    --validation_prompt_dir /scratch/10320/lanqing001/xinrui/FastVideo/data/Inte4K/validation \
    --gradient_checkpointing \
    --train_batch_size=1 \
    --num_latent_t 12 \
    --sp_size 2 \
    --train_sp_batch_size 1 \
    --fsdp_sharding_startegy "full" \
    --dataloader_num_workers 16 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=1e-5 \
    --mixed_precision=bf16 \
    --checkpointing_steps=2 \
    --validation_steps 2000 \
    --validation_sampling_steps 50 \
    --checkpoints_total_limit 5 \
    --allow_tf32 \
    --ema_start_step 0 \
    --cfg 0.0 \
    --ema_decay 0.999 \
    --log_validation \
    --output_dir=/scratch/10320/lanqing001/xinrui/FastVideo/data/outputs/ZeroInit2_blocks_10_gpus_8_bs_2 \
    --tracker_project_name GH200_Hunyuan_ControlNet_Finetune_ZeroInit2 \
    --num_frames 45 \
    --num_height 1088 \
    --num_width 1920 \
    --shift 7 \
    --validation_guidance_scale "1.0"


EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
        echo "Job failed with exit code $EXIT_CODE. See job.${SLURM_JOB_ID}.err for details."
        exit $EXIT_CODE
else
        echo "Job completed successfully."
fi
