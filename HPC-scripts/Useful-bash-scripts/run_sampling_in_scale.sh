#!/bin/bash
for dir in scale=*; do
folder="$dir"
 cd $folder
 

for shear in shear=*; do
	cd "$shear"
	sbatch RunSampling.sh
	cd ..
done

cd ..
 

done
