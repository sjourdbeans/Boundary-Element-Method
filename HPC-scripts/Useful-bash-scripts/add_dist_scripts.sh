#!/bin/bash

for rate in "$@"; do
  folder="shear=$rate"


  cp ../../../../RunSampling.sh "$folder"/
  #cp CalcDistributions.py "$folder"/
  cp SampleOrientationResults.py "$folder"/

  sed -i "s/^#SBATCH --job-name=.*/#SBATCH --job-name=\"${rate}_shear_sample\"/" "$folder/RunSampling.sh"

  sed -i "s/^shear_rate *= *.*/shear_rate = ${rate}/" "$folder/SampleOrientationResults.py"

done
