function [collocpts,Zs, areas] = gencolloc(panels)
% Calculates a list of collocation points by computing panel centroids.
% Panels are stored as 3-D array
% [panel verts, cond num, 0]
% [vert 1 x,y,z]
% [vert 2 x,y,z]
% [vert 3 x,y,z]
% [Other stuff]

% [vert n x,y,z]
 collocpts(1,3) = 0;
 [rows,cols,numpanels] = size(panels);
 for i=1:numpanels
   numverts = panels(1,1,i);
   panel = panels(2:numverts+1,:,i);
   [area, collocpt,Z] = calcpStokes(panel);
   collocpts(i,:) = collocpt;
   Zs(i,:) = Z;
   areas(i) = area;
 end

