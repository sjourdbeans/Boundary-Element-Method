function [panels] = readpanels(file)
% Read the file, line by line, and dispatch based on first character on line.
% Panel type, cond number, x1, y1, z1, x2, y2, z2, ... xn, yn, zn
% Panel type:
% Q means quadralateral
% T means triangle
% 0 or # or * means a comment
% Conductor number:
% An integer indicting a conductor number.
% Panel is stored as 3-D array
% [panel verts, cond num, 0]
% [vert 1 x,y,z]
% [vert 2 x,y,z]
% [vert 3 x,y,z]
%
%
% [vert n x,y,z]
% [potential, d/dn potential, type]


panels(1,1,1) = 0.0;

fid = fopen(file, 'r');
while 1
  line = fgetl(fid);
  if ~ischar(line), break, end
  if length(line) ~= 0, 
    if (strcmp(line(1),'Q') > 0) | (strcmp(line(1),'q') > 0) 
      panels = readpanel(line, panels, 4);
    elseif(strcmp(line(1),'T') > 0) | (strcmp(line(1),'t') > 0) 
      panels = readpanel(line, panels, 3);
    elseif((strcmp(line(1),'0') == 0) & (strcmp(line(1),'#') == 0) & (strcmp(line(1),'*') == 0));
      'syntax error in input file'
      line
    end
  end
end

% Get read of the dummy header panel.
[r,c,numpanels] = size(panels);
panels = panels(:,:,2:numpanels);


