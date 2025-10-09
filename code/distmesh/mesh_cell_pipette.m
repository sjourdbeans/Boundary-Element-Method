clear all; close all; clc
%%%%%%% Generate surface mesh for what we need %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
user = 'Waveform';
switch user
    case 'Da'
        cd('D:\004 FLOW VELOCIMETRY DATA\')
        run('D:\002 MATLAB codes\000 Routine\subunit of img treatment routines\add_path_for_all.m')
        % will add all the pathes needed
    case 'Daniel'
    case 'Waveform'
        cd('M:\tnw\bn\mea\Shared\Algae\005 Flow velocimetry images')
    case 'other'
        %Please fill in the relevant folders here
end
topfld = 'M:\tnw\bn\mea\Shared\Algae\005 Flow velocimetry images';
cd(topfld) % so that the pipette file will be saved parallel to the folder 
             % of the full scenario.

%%
experiment = '190224c47wvc';
AA05_experiment_based_parameter_setting;          
pipettefile    ='pipette_20190224_c47wvc';            
             
%% Initial guess for the mesh
load('pipette_20180504_c18r.mat','-regexp','^(?!(a|b|d)$).') 
% load an initial pipette. % pipette_20180504_c18r

%% Define pipette 
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultAxesFontSize',16)
N = 10;   theta  = 0:(pi/2)/N:(pi/2);
N = 20;   thetah = 0:(pi)/N:(pi);
Nt = 5;

alpha = 4*pi/180;   % pipette angle 
ta    = tan(alpha);
NL    = 15;
L     = NL*a; % length of the pipette
trans_para = 4;
% Furthest point of pipette.
Rp    = ta*L;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Circular cap:
p1 = [(Rp+trans_para*a*ta)*cos(theta(:))+L (Rp+trans_para*a*ta)*sin(theta(:))];
p1 = p1(1:(end-1),:);

% Pipette 
p2 = [a*(NL:-1:2)' ta*a*(NL:-1:2)'+trans_para*a*ta];

% Head
x  = a*cos(thetah); y = b*sin(thetah);
ind = find(y(:)-d > 0);
p3 = [x(ind(1):end)' y(ind(1):end)']; 

% head pipette transition
xt = ((p2(end,1)+a/(2*Nt)):(-a/Nt):(p3(1,1)-a/(2*Nt)))';
yt = spline([p1(:,1);p2(:,1);p3(:,1)],[p1(:,2);p2(:,2);p3(:,2)],xt);

% putting everything together:
pv = [p1;p2;xt yt;p3];
pv = [pv ; pv((end-1):-1:1,1) -pv((end-1):-1:1,2)];
% pv = [x' y'];

plot(pv(:,1),pv(:,2),'-'); axis equal;
pause

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute mesh
fh=@(p) 0.05+0.5*dsphere(p,0,0,1,0);

fh    = @(p) 1.7+4.0*(dsphere(p,0,0,0,0))/L;
fd    = @(p) dPipette(p,pv);

[p,t] = distmeshsurface(fd,fh,2.0,1.7*[-a,-Rp,-Rp;L+Rp,Rp,Rp],p,t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save in the appropriate format, as .qif and .mat

qiffile = sprintf('%s.qif',pipettefile);
matfile= sprintf('%s.mat',pipettefile);

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
% 
% % then save the .mat file
[panels] = readpanels(qiffile);  
[nx ny nz] = size(panels);
done = 'read input file'
fclose(fid);

plotpanels(panels,ones(nz,1))
view(3)
axis equal
xlabel('x [$\mu$m]'),ylabel('y [$\mu$m]'),zlabel('z [$\mu$m]')
% cleanfigure; 
% matlab2tikz('Mesh_pipette.tex','extraAxisOptions','scale=\figurescale');
cameratoolbar

save(matfile,'a','b','d','p','t','pv','panels');






