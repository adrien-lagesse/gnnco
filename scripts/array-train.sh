#!/bin/bash

#SBATCH --job-name=GNNCO                                                                   # create a short name for your job
#SBATCH --nodes=1                                                                          # node count
#SBATCH --partition=gpu                                                                    # Name of the partition
#SBATCH --gres=gpu:1                                                                       # Require GPU
#SBATCH --mem-per-gpu=20G                                                                
#SBATCH --time=47:00:00                                                                    # total run time limit (HH:MM:SS)
#SBATCH --mincpus=8                                                                        # Number of CPUS
#SBATCH --output="/home/jlagesse/gnnco/sbatch-output/%x   %j.out"                          # Output File

set -x
cd ${SLURM_SUBMIT_DIR}

noises=(0.02 0.08 0.12 0.18 0.24 0.3 0.35)

echo ${noises[${SLURM_ARRAY_TASK_ID}]}

rye run gm-train \
    --dataset  "/home/jlagesse/gnnco/data/PCQM4Mv2[${noises[${SLURM_ARRAY_TASK_ID}]}]" \
    --experiment "PCQM4Mv2" \
    --run-name "GAT-Large PCQM4Mv2[${noises[${SLURM_ARRAY_TASK_ID}]}]" \
    --epochs 500 \
    --batch-size 1000 \
    --cuda \
    --log-frequency 25 \
    --profile \
    --model GAT \
        --layers 6 \
        --heads 12 \
        --features 240 \
        --out-features 64 \
    --optimizer adam-one-cycle \
        --max-lr 3e-3 \
        --start-factor 5 \
        --end-factor 500 \
        --grad-clip 0.1