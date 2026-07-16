#!/bin/bash

for dir in shear=*; do
  if [ -d "$dir" ]; then
    (
      cd "$dir" || exit
      echo "Submitting in $dir"
     # sbatch RunTrajectorySim.sh
      sbatch RunSampling.sh

    )
  fi
done
