clear all;
close all;
%%%%%%% Generate surface mesh for what we need %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define testing points
FILE = 'spheroid_05';
qiffile = sprintf('%s.qif',FILE);
matfile= sprintf('%s.mat',FILE);

[X,Y,Z] = ndgrid(-500:10:500,-500:10:500,0);

% Define pipette 
N = 10;   theta  = 0:(pi/2)/N:(pi/2);
N = 20;   thetah = 0:(pi)/N:(pi);
Nt = 10;

a     = 5;   % half major axis of prolate spheroid
b     = 2.5;   % half minor axis of prolate spheroid

% Head
x  = a*cos(thetah); y = b*sin(thetah);
p3 = [x(:) y(:)]; 

pv = p3;
pv = [pv ; pv((end-1):-1:1,1) -pv((end-1):-1:1,2)];
% pv = [x' y'];

plot(pv(:,1),pv(:,2),'-'); axis equal;
pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute mesh
fh=@(p) 0.05+0.5*dsphere(p,0,0,1,0);

fh    = @(p) 1.7+4.0*(dsphere(p,0,0,0,0))/L;
fd    = @(p) dPipette(p,pv);

[p,t] = distmeshsurface(fd,@huniform,0.4,1.1*a*[-1,-1,-1;1,1,1],[],[]);
% [p,t] = distmeshsurface(fd,fh,2.0,1.7*[-a,-Rp,-Rp;L+Rp,Rp,Rp],p,t);

% load('pipette_02.mat');
% [p,t] = distmeshsurface(fd,fh,1.7,1.4*[-a,-Rp,-Rp;L+Rp,Rp,Rp],[],[]);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save in the appropriate format, as .qif and .mat

% First save the .qif
npanel = size(t,1);

fid =  fopen(qiffile,'w');
fprintf(fid,'0 sphere with %d. \n',npanel);

for i = 1:npanel
    fprintf(fid,'T  1 ');   
    fprintf(fid,'%3.15f ',p(t(i,1),:));
    fprintf(fid,'%3.15f ',p(t(i,2),:));
    fprintf(fid,'%3.15f ',p(t(i,3),:));
    fprintf(fid,'\n');    
end
fclose(fid);

% then save the .mat file
[panels] = readpanels(qiffile);  
[nx ny nz] = size(panels);
done = 'read input file'

plotpanels(panels,ones(nz,1))
view(3)
axis equal
cameramenu

save(matfile,'p','t','pv','panels');






