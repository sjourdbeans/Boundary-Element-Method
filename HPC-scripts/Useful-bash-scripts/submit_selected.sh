#!/bin/bash

for rate in "$@"; do
  dir="shear=$rate"

  if [ -d "$dir" ]; then
    (
      cd "$dir" || exit
      echo "Submitting in $dir"
      sbatch RunSampling.sh
      # sbatch RunTrajectorySim.sh
    )
  else
    echo "Folder $dir does not exist"
  fi
done
