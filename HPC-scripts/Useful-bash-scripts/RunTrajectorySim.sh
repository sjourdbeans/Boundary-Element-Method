#!/bin/bash

#SBATCH --job-name="0_shear_traj"
#SBATCH --time=16:00:00
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=6
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=3GB
#SBATCH --account=research-me-pe

module load 2025
module load openmpi


filepath="/home/sbuitjes/code/Boundary-Element-Method"

srun "$filepath/.venv/bin/python" "CalcOrientations.py" > Trajectories.log


