#!/bin/bash
#SBATCH --mem=32g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA100x8      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bcjw-delta-gpu    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=finetuning-llama3-8b-instruct
#SBATCH --time=02:00:00      # hh:mm:ss for the job
#SBATCH --gpus-per-node=8

source /u/boqiny2/miniconda3/etc/profile.d/conda.sh 
conda activate /u/boqiny2/miniconda3/envs/swift
bash examples/pytorch/llm/scripts/llama3_8b_instruct/lora/sft.sh