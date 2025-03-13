#!/bin/bash
#SBATCH --job-name=lanqing001
#SBATCH -A CGAI24022
#SBATCH --nodes=8               # Request 8 nodes
#SBATCH --ntasks-per-node=1     # Ensure 1 task per node
#SBATCH --time=30:00:00         # Max runtime
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


# Base model path
export MODEL_BASE=/scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan/

# Checkpoint directory (containing model_shard_X.pth files)
CHECKPOINT_DIR="/scratch/10320/lanqing001/xinrui/FastVideo/data/outputs/ZeroInit2_blocks_10_gpus_8_bs_2_step_5k_2/checkpoint-2000"

# Create output directory if it doesn't exist
mkdir -p /scratch/10320/lanqing001/xinrui/FastVideo/outputs_video/hunyuan_controlnet/ZeroInit2_blocks_10_gpus_8_bs_2_step_5k_2/2000/

# Run the inference directly with srun
srun -N 8 -n 8 \
     python /scratch/10320/lanqing001/xinrui/FastVideo/fastvideo/sample/sample_controlnet_TACC.py \
     --height 720 \
     --width 1280 \
     --num_frames 45 \
     --num_inference_steps 50 \
     --guidance_scale 1 \
     --embedded_cfg_scale 6 \
     --fps 8 \
     --pretrained_model_name_or_path /scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan \
     --dit_model_name_or_path /scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
     --flow_shift 7 \
     --flow-reverse \
     --prompt /scratch/10320/lanqing001/xinrui/FastVideo/assets/prompt_1-2.txt \
     --seed 1024 \
     --output_path /scratch/10320/lanqing001/xinrui/FastVideo/outputs_video/hunyuan_controlnet/ZeroInit2_blocks_10_gpus_8_bs_2_step_5k_2/2000/output.mp4 \
     --model_path /scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan \
     --dit-weight ${CHECKPOINT_DIR} \
     --vae-sp \
     --model_type hunyuan_controlnet \
     --precision bf16

echo "Inference completed. Check output directory for generated videos."

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
        echo "Job failed with exit code $EXIT_CODE. See job.${SLURM_JOB_ID}.err for details."
        exit $EXIT_CODE
else
        echo "Job completed successfully."
fi
