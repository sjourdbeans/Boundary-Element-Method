# Boundary Element Method

## Note: the documentation of this repository is not complete. More useful scripts will be added at a later time

This is a repository which includes a boundary element method (BEM) code to compute the flow around on and around a microswimmer. This code solves the double layer density integral equation of the second kind described by Power and Miranda with the completion flow of Keaveny and Shelley and includes the flaggella using Slender Body Theory.

To use this code, a mesh is needed. In the original [code](https://github.com/DidasW/Matlab_TUD) a mesh was generated with distmesh and it included a cell body and a pipette. However, for this code the goal is to model a free microswimmer, so we do not need the pipette. 

To generate the mesh, run the python file `mesher.py` with the right settings (e.g. length, width, elements, etc.) and make sure to change the directory where the mesh will be saved. 

The source code of the solver can be found in the src folder. However, the code can be installed using `uv` or `pip` and can then be called using:

`import bemsolver as bem`

Most of the documentation is already in the source code, but for clarity the documentation will be updated here.



