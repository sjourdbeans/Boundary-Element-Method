% Prolate spheroid: x^2/a^2 + y^2/a^2 + z^2/c^2 = 1  (with c > a)

a  = 5;              % equatorial semi-axis
b  = 1;              % polar semi-axis
% b_arr= 0.6:0.05:1;
h0 = 0.2;             % target edge length (surface resolution)


% for j=1:length(b_arr)

% b=b_arr(j);

pv = spheroid_profile(a,b,101);

file_path="/home/sjoerd-buitjes/University/Master-Thesis/BEM/Boundary-Element-Method/datafiles/mesh/jeffery-orbits-fine/";
matfile =fullfile(file_path,sprintf('spheroid_mesh_b=%d.mat', b));
qiffile = fullfile(file_path, 'qif', sprintf('spheroid_mesh_b=%d.qif', b));

% --- signed distance function (zero on surface) ---
fd = @(p) sqrt((p(:,1)/b).^2 + (p(:,2)/a).^2 + (p(:,3)/a).^2) - 1;

% --- size function (uniform); tweak for denser poles if desired (see below) ---
fh = @(p) h0 + 0*p(:,1);
% fh = @(p) max(0.4*h0, h0*(1 - 0.7*min(1, abs(p(:,1))/a)));


% --- bounding box and fixed points (none needed) ---
bbox = [-a, -a, -b;  a,  a,  b];
pfix = [];

% --- projection onto the spheroid surface (radial snap) ---
proj = @(p) projspheroid(p,a,b);

% --- run surface mesher ---
[p,t] = distmeshsurface(fd, fh, h0, bbox, pfix, proj);

% After: [p,t] = distmeshsurface(fd, fh, h0, bbox, pfix, proj);

% (1) make sure all points lie exactly on the spheroid
p = proj(p);   % one last snap

% (2) consistently orient all triangles to point outward
c = (p(t(:,1),:) + p(t(:,2),:) + p(t(:,3),:)) / 3;   % triangle centroids
n = cross(p(t(:,2),:) - p(t(:,1),:), p(t(:,3),:) - p(t(:,1),:), 2); % face normals

% outward test: dot(normal, position-from-center) should be > 0
out = sum(n .* c, 2) < 0;
flip = find(~out);
t(flip,[2 3]) = t(flip,[3 2]);   % swap two vertices to flip the normal

% qiffile = sprintf(spheroid_qif);
% matfile= sprintf(spheroid_mesh);

% First save the .qifqiffile = sprintf('%s.qif',spheroid_qif);

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
[nx, ny, nz] = size(panels);
done = 'read input file'


plotpanels(panels,ones(nz,1))
view(3)
axis equal
xlabel('x [$\mu$m]'),ylabel('y [$\mu$m]'),zlabel('z [$\mu$m]')
cleanfigure; 
% matlab2tikz('Mesh_pipette.tex','extraAxisOptions','scale=\figurescale');
cameratoolbar

save(matfile,'a','b','d','p','t','pv','panels');

% end
% visualize
% trisurf(t, p(:,1), p(:,2), p(:,3), 'FaceColor',[0.8 0.9 1], 'EdgeColor',[0.3 0.3 0.3]);
% axis equal vis3d; camlight; lighting gouraud;
% title('Prolate spheroid (surface mesh)');



function p = projspheroid(p,a,b)
x = p(:,1); y = p(:,2); z = p(:,3);
s = sqrt((x./a).^2 + (y./b).^2 + (z./b).^2);
mask = s > 0;
x(mask) = x(mask)./s(mask);  y(mask) = y(mask)./s(mask);  z(mask) = z(mask)./s(mask);
x(~mask) = 0; y(~mask) = 0; z(~mask) = b; % send exact origin to a pole
p = [x,y,z];
end


function pv = spheroid_profile(a,b,N)
% pv.x are axial positions (length N) from -a to a
% pv.r are the corresponding cross-sectional radii
% Also returns [x r] as pv.array for convenience.

if nargin < 3, N = 201; end       % resolution along x
x = linspace(-a, a, N).';
r = b * sqrt(max(0, 1 - (x./a).^2));   % clamp for safety

% pv = x;
% pv.r = r;
pv = [x r];

% Optional: derivatives if you need normals/curvature/spacing control
% pv.dr_dx = -(b/a^2) * x ./ max(sqrt(1 - (x./a).^2), eps);

% % (Optional) quick checks:
% pv.volume = (4/3)*pi*a*b^2;                 % exact volume
% e = sqrt(1 - (b/a)^2);                      % eccentricity (prolate)
% pv.area  = 2*pi*b^2*(1 + (a/(b*e))*asin(e)); % exact surface area
end
