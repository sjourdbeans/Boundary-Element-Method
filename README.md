# Boundary Element Method

This is a repository which includes a boundary element method (BEM) code to compute the flow around on and around a microswimmer. To use this code, a mesh is needed. In the original [code](https://github.com/DidasW/Matlab_TUD) a mesh was generated with distmesh and it included a cell body and a pipette. However, for this code the goal is to model a free microswimmer, so we do not need the pipette. \\

To generate the mesh, run the python file `mesher.py` with the right settings (e.g. length, width, elements, etc.) and make sure to change the directory where the mesh will be saved. \\

Go to `/code/initialisation/initialise_files.m` and adjust it to your working directory and change the parameters such that the right files are selected for the waveform and the mesh. Furthermore, set the filelocations to save the results. This file makes sure that the matlab files in the different folders are added to the path such that they can be called from matlab files in other folders. \\

Once that is done, the file `/code/initialisation/initialise_mesh.m` can be run for a given backrground flow. A plot of the double layer potential on your will appear. \\

To run the full simulation, adjust the parameters in the beginning of `main.m` for your case. **NOTE: It is important that the background flow set in `initialise_mesh.m` is the same background flow set in `main.m`!** Once that is done, you can run `main.m`.



