#!/bin/bash

for scale in "$@"; do
  folder="scale=$scale"
  mkdir -p "$folder"

  cp CalcOrientations.py "$folder"/
  cp SampleOrientationResults.py "$folder"/
  cp RunTrajectorySim.sh "$folder"/
  cp RunSampling.sh "$folder"/
 # cp CalcDistributions.py "$folder"/
  cp make_shear_folders.sh "$folder"/
  cp submit* "$folder"/


  sed -i "s/^scale_out_of_plane *= *.*/scale_out_of_plane = ${scale}/" "$folder/CalcOrientations.py"
#  sed -i "s/^scale_out_of_plane *= *.*/scale_out_of_plane = ${scale}/" "$folder/CalcDistributions.py"
  sed -i "s/^scale_out_of_plane *= *.*/scale_out_of_plane = ${scale}/" "$folder/SampleOrientationResults.py"



  chmod +x "$folder/make_shear_folders.sh"
done
