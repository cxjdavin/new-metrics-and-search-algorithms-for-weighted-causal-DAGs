#!/bin/sh
#SBATCH --partition=long
#SBATCH --time=4320

srun script-setup.sh

for i in 1 2 3 4 5
do
    echo ${i}
    sbatch slurm-run.sh ${i}
done

