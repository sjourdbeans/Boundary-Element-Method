# Boundary Element Method

This is a repository which includes a boundary element method (BEM) code to compute the flow around on and around a microswimmer. This code solves the double layer potential integral equation of the second kind described by Keaveny and Shelley and includes the flaggella using Slender Body Theory.

 To use this code, a mesh is needed. In the original [code](https://github.com/DidasW/Matlab_TUD) a mesh was generated with distmesh and it included a cell body and a pipette. However, for this code the goal is to model a free microswimmer, so we do not need the pipette. 

To generate the mesh, run the python file `mesher.py` with the right settings (e.g. length, width, elements, etc.) and make sure to change the directory where the mesh will be saved. 

## Update README.md for python code



