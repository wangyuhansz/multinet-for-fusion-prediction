#!/bin/bash

#SBATCH -J tl_gpu
#SBATCH -p debuga100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

module load miniconda3/4.10.3 cuda
source activate multinet
which python
python main.py