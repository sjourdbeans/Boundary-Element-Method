function [MATRIX,LINE_S,LINE_R,COLN_S,COLN_R] = FlowStokes(panels,evalpoints,~,p)
% Fills in matrix relating panel charges to collocation point potentials.

% Setting up the evaluation Matrix, hence the singularity in the computation of all the integral is not removed 
% (integral involves a normal surface vector dot position vector NOT on surface and is non zero and non singular!)

% Setting up the collocation matrix
 [~,~,numpanels] = size(panels);
 [npoints,~]       = size(evalpoints);
 MATRIX   = [];                         %Mobility matrix?
 LINE_S   = [];                         %Line Stokeslet?
 LINE_R   = [];                         %Line Rotlet?
 COLN_S   = zeros(3*npoints,3);
 COLN_R   = zeros(3*npoints,3);
 
 % Loop through the boundary elements. (Matrix is formed by row, correponding 
 % the contribution of each element to the integral equations written at 
 % each collocation point)
 
 for i=1:numpanels
   if rem(i,100)==0  
   fprintf('Computing panel %d out of %d \n',i,numpanels);
   end
   numverts = panels(1,1,i);
   panel = panels(2:numverts+1,:,i);
   [~, ~, ~,FD,LS,LR,~] = calcpStokes(panel,evalpoints,p);
   MATRIX   = [MATRIX    FD];
   LINE_S   = [LINE_S    LS];  
   LINE_R   = [LINE_R    LR];
 end