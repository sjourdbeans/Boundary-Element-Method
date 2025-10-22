function [MATRIX,LINE_S,LINE_R,COLN_S,COLN_R,FSS,identity,C_R] = collocationStokes(panels,centroids,normals,p,colloc)
% Fills in matrix relating panel charges to collocation point potentials.
% Panel is stored as 3-D array
% [panel verts, cond num, 0]
% [vert 1 x,y,z]
% [vert 2 x,y,z]
% [vert 3 x,y,z]
%
%
% [vert verts x,y,z]
% Centroids stored as a matrix with 3 columns.
% [vert 1 x,y,z]
% [vert 2 x,y,z]
% [vert 3 x,y,z]
%
%

% Setting up the collocation Matrix, hence the singularity in the computation of all the integral is removed 
% (integral involves a normal surface vector dot position vector on surface and is hence zero)
% This is done with colloc == 1;

% Setting up the collocation matrix
 [rows,cols,numpanels] = size(panels);
 MATRIX     = [];
 LINE_S   = [];
 LINE_R   = [];
 FSS =[];
%  LINE_R =zeros(3*numpanels,3);
 COLN_S   = zeros(3*numpanels,3);
 COLN_R   = zeros(3*numpanels,3);
 identity =[];

 C_R=[];
%  FSS=0;

 % Loop through the boundary elements. (Matrix is formed by row, ]
 % correponding the the contribution of each element to the integral equations writte at each collocation point)
 
 Surf = 0;
 for i=1:numpanels   
   if rem(i,10)==0  
   fprintf('computing panel %d out of %d \n',i,numpanels);  
   end
   numverts = panels(1,1,i);
   panel = panels(2:numverts+1,:,i);
   [area, collocpt, Z,FD,LS,LR,S,C_r] = calcpStokes(panel, centroids ,p);
   MATRIX     = [MATRIX      FD];
   LINE_S   = [LINE_S    LS];  
   LINE_R   = [LINE_R LR];
   C_R= [C_R C_r];
   FSS = [FSS S];
  %  Surf     = Surf + S;
   identity = [identity eye(3)];

 end

MATRIX = 0.5*eye(3*numpanels) + MATRIX + FSS +C_R; 

