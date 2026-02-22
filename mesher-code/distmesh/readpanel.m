function [List] = readpanel(line,List,panelsize)
% Reads a panel with vertices as floating point numbers on a line

% Hack below deals with this programmer's limited ability to figure
% out matlab's approach to handling lines with mixed text and numbers.
[stuff,cnt]=sscanf(line,'%s %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f');
   if (cnt == (2 + panelsize * 3)) 
     fastcap = 1;
     fastlap = 0;
   elseif (cnt == (2 + 3 + panelsize * 3)) 
     fastcap = 0;
     fastlap = 1;
   else
     disp('Format error in panel!');
     return;
   end

% Stuff now has the vertices of a panel. 
% Stick in List (a multidimensional array).
   [rows,cols,numpanels] = size(List);
   numpanels = numpanels +1; 
   List(1,1,numpanels) = panelsize;
   List(1,2,numpanels) = stuff(2);
   List(1,3,numpanels) = 0;
   if(fastlap == 1) 
     List(panelsize+2,1,numpanels) = stuff(cnt - 2);  % Potential
     List(panelsize+2,2,numpanels) = stuff(cnt - 1);  % d/dn Potential
     List(panelsize+2,3,numpanels) = stuff(cnt);      % Type
   end;
   index = 3;
   for i=2:panelsize+1
     for j=1:3
       List(i,j,numpanels) = stuff(index);
       index = index + 1;
     end
   end




