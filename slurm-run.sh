#!/bin/sh
#SBATCH --partition=long
#SBATCH --time=4320

srun script$1.sh

