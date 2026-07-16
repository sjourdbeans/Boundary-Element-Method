#!/bin/bash

for rate in "$@"; do
  folder="shear=$rate"
  mkdir -p "$folder"

  cp CalcOrientations.py "$folder"/
  cp RunTrajectorySim.sh "$folder"/
  cp RunSampling.sh "$folder"/
  #  cp CalcDistributions.py "$folder"/
  cp SampleOrientationResults.py "$folder"/
  sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"${rate}_shear_sym\"/" "$folder/RunTrajectorySim.sh"
  sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"${rate}_shear_sample\"/" "$folder/RunSampling.sh"

  sed -i "s/^shear_rate *= *.*/shear_rate = ${rate}/" "$folder/CalcOrientations.py"
#  sed -i "s/^shear_rate *= *.*/shear_rate = ${rate}/" "$folder/CalcDistributions.py"
  sed -i "s/^shear_rate *= *.*/shear_rate = ${rate}/" "$folder/SampleOrientationResults.py"


  chmod +x "$folder/RunTrajectorySim.sh"
done
