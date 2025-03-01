#!/bin/bash
#SBATCH --job-name=lanqing001
#SBATCH -A CGAI24022
#SBATCH --nodes=4               # Request 4 nodes
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

srun -N 4 -n 4 \
    python /scratch/10320/lanqing001/xinrui/FastVideo/data_preprocess/preprocess_vae_latents.py \
    --model_path "/scratch/10320/lanqing001/xinrui/FastVideo/data/hunyuan" \
    --data_merge_path "/scratch/10320/lanqing001/xinrui/FastVideo/data/raw_mp4_10k_720/merge.txt" \
    --train_batch_size=8 \
    --max_height=720 \
    --max_width=1080 \
    --num_frames=45 \
    --dataloader_num_workers 1 \
    --output_dir="/scratch/10320/lanqing001/xinrui/FastVideo/data/raw_mp4_10k_1088/" \
    --model_type "hunyuan" \
    --train_fps 8


EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
        echo "Job failed with exit code $EXIT_CODE. See job.${SLURM_JOB_ID}.err for details."
        exit $EXIT_CODE
else
        echo "Job completed successfully."
fi
