#!/bin/bash

#SBATCH --job-name="CreateSwimmerObject"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --partition=compute
#SBATCH --mem-per-cpu=3GB
#SBATCH --account=research-me-pe


module load 2025
module load openmpi


filepath="/home/sbuitjes/code/Boundary-Element-Method"

srun "$filepath/.venv/bin/python" "CreateSwimmerObject.py" > swimmer.log
# srun "$filepath/.venv/bin/python" "CreateScaledOutOfPlane.py" > swimmer.log
#srun "$filepath/.venv/bin/python" "CreateSymmetricSwimmer.py" > swimmer.log
# srun "$filepath/.venv/bin/python" "CreateEuglenaObject.py" > swimmer.log
#srun "$filepath/.venv/bin/python" "CreateMultipleScaleOOP.py" > swimmer.log

