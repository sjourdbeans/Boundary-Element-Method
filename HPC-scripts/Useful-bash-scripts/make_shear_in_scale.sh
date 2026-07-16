#!/bin/bash
for dir in scale=*; do
folder="$dir"
 cd $folder
 
 ./make_shear_folders.sh "$@"
for shear in "$@"; do
	cd "shear=$shear"
	sbatch RunTrajectorySim.sh
	cd ..
done

cd ..
 

done
