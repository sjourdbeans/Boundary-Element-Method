clear all;
close all;
%%%%%%% Generate surface mesh for what we need %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define testing points
FILE = 'pipette_14';
qiffile = sprintf('%s.qif',FILE);
matfile= sprintf('%s.mat',FILE);

[X,Y,Z] = ndgrid(-500:10:500,-500:10:500,0);
p       = [X(:) Y(:) Z(:)];

% Define pipette 
N = 10;   theta  = 0:(pi/2)/N:(pi/2);
N = 20;   thetah = 0:(pi)/N:(pi);
Nt = 10;

a     = 5;   % half major axis of prolate spheroid
b     = 3.7;   % half minor axis of prolate spheroid
d     = 2.8;   % size of the pipette opening
alpha = 5.5*pi/180;   % pipette angle (8 degrees)
ta    = tan(alpha);
NL    = 10;
L     = NL*a; % length of the pipette

% Furtherst point of pipette.
Rp    = ta*L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Circular cap:
p1 = [Rp*cos(theta(:))+L Rp*sin(theta(:))+2*a*ta];
p1 = p1(1:(end-1),:);

% Pipette 
p2 = [a*(NL:-1:3)' ta*a*(NL:-1:3)'+2*a*ta];

% Head
x  = a*cos(thetah); y = b*sin(thetah);
ind = find(y(:)-d > 0);
p3 = [x(ind(1):end)' y(ind(1):end)']; 

% head pipette transition
xt = ((p2(end,1)+a/(2*Nt)):(-a/Nt):(p3(1,1)-a/(2*Nt)))';
yt = spline([p1(:,1);p2(:,1);p3(:,1)],[p1(:,2);p2(:,2);p3(:,2)],xt);

% putting everything togetehr:
pv = [p1;p2;xt yt;p3];
pv = [pv ; pv((end-1):-1:1,1) -pv((end-1):-1:1,2)];

plot(pv(:,1),pv(:,2),'-'); axis equal;
pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute mesh

fh    = @(p) 1.4+20/15*3.5*(dsphere(p,a,0,0,0)/L);
fd    = @(p) dPipette(p,pv);
[p,t] = distmeshsurface(fd,@huniform,1.4,1.4*[-a,-Rp,-Rp;L+Rp,Rp,Rp],[],[]);

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






