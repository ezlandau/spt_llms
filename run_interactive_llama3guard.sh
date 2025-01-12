#!/bin/bash

#SBATCH --job-name=interactive-llama3guard
#SBATCH --output=/ukp-storage-1/zadorin/out.txt
#SBATCH --error=/ukp-storage-1/zadorin/error.txt
#SBATCH --mail-user=egor.zadorin@stud.tu-darmstadt.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT,INVALID_DEPEND
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --nodelist=penelope

source /ukp-storage-1/zadorin/.bashrc

export HF_TOKEN="HF_TOKEN"

huggingface-cli login --token $HF_TOKEN

echo "Starting vLLM server with Meta LLaMA Guard 3 8B..."
vllm serve --device cuda meta-llama/Llama-Guard-3-8B

# Keep the script running
wait
