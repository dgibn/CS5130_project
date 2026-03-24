#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=Qnet
#SBATCH --partition=jiang
#SBATCH --nodes=1
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=12
#SBATCH --output=/projects/vig/divs/CS5130_project/output/train3.log
#SBATCH --error=/projects/vig/divs/CS5130_project/output/train3.log

CUDA_VISIBLE_DEVICES="0" python3 /projects/vig/divs/CS5130_project/src/train.py