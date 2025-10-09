close all;
clear all;
FILE = 'spheroid_00';
FILE = 'ellipse_03';

infile = sprintf('%s.mat',FILE);
outfile= sprintf('%s.qif',FILE);

load(infile)

for i = 1:nz
    x                = panels(2:nx,1,i);
    P1               = x;
    P2               = 0.5*(3*x.^2-1);
    P3               = 0.5*(5*x.^3-3*x);
    P4               = 0.125*(35*x.^4-30*x.^2+3);
    y                = max(sqrt(panels(2:nx,2,i).^2+panels(2:nx,3,i).^2),10e-8);
    r                = max(b*(1-x.^2).*(1+a1*P1+a2*P2+a3*P3+a4*P4),10e-8);
    panels(2:nx,1,i) =    panels(2:nx,1,i); 
    panels(2:nx,2,i) = (r./y).*panels(2:nx,2,i); 
    panels(2:nx,3,i) = (r./y).*panels(2:nx,3,i);
end

% Resize
npanel = size(t,1);

fid =  fopen(outfile,'w');
fprintf(fid,'0 sphere with %d. \n',npanel);

for i = 1:npanel
    fprintf(fid,'T  1 ');   
    fprintf(fid,'%3.15f ',p(t(i,1),:));
    fprintf(fid,'%3.15f ',p(t(i,2),:));
    fprintf(fid,'%3.15f ',p(t(i,3),:));
    fprintf(fid,'\n');    
end
fclose(fid);

[panels] = readpanels(outfile);  
[nx ny nz] = size(panels)
done = 'read input file'

plotpanels(panels,ones(nz,1))
view(3)
axis equal
cameramenu