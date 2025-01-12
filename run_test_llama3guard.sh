#!/bin/bash

#SBATCH --job-name=llama_guard_eval
#SBATCH --output=/ukp-storage-1/zadorin/llama_guard_eval_out.txt
#SBATCH --error=/ukp-storage-1/zadorin/llama_guard_eval_err.txt
#SBATCH --mail-user=egor.zadorin@stud.tu-darmstadt.de
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,STAGE_OUT,INVALID_DEPEND
#SBATCH --ntasks=1
#SBATCH --constraint="gpu_model:a100"
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/zadorin/.bashrc

srun python /ukp-storage-1/zadorin/spt_llms/evaluations/classify_completions_llamaguard.py
