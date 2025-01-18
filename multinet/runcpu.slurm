#!/bin/bash

#SBATCH -J tl_cpu
#SBATCH -p 64c512g
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

module load miniconda3/4.10.3
source activate multinet
which python
python main.py