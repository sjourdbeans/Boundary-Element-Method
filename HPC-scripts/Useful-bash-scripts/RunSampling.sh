#!/bin/bash

#SBATCH --job-name="0_shear_Sample"
#SBATCH --time=00:04:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=3GB
#SBATCH --account=research-me-pe

module load 2025
module load openmpi


filepath="/home/sbuitjes/code/Boundary-Element-Method"

srun "$filepath/.venv/bin/python" "SampleOrientationResults.py" > Sample.log


