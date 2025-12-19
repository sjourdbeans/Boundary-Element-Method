U(1) = 1;               % U is the velocity translation
U(2) = 0;
U(3) = 0; 

W(1) = 0;               % W is the rotation velocity
W(2) = 0;
W(3) = 0;


load(waveformsfile); % File with waveforms

rf         = 0.2;                   % Radius of the flagellum, [micron]...           
                                     % 0.20+/-0.03, TEM Calib...
                                     % @20170601 

% lf         = lf0;                    % Length of the flagellum  [micron]
lf =10;
Slend      = rf/lf;                  % Slenderness ratio        [-]
kappasave(:,:,2:end) = kappasave(:,:,2:end)./lf;% Rescale curvature in ...
Rotmat = @(phi) [cos(phi) sin(phi); -sin(phi) cos(phi);];         % Rotation matrix

velx       = velx0.*lf;                         %               [micron/s]
vely       = vely0.*lf;                         %               [micron/s]
% Nf         = size(kappasave,3)-2;     % # of flagella elements  [-]

curv = zeros(3,1);

Nf=length(curv) - 1;

ssold      = linspace(0,lf,Nf+1);     % Flagella segments       [micron]
smin       = 0 ;%0.15;                    % Flagellum starts at...  [-]
                                      % smin(0-1) of the total length  
                                      % Rationale behind:
                                      % To avoid collision between points 
                                      % on flagella and those on the cell. 

indstart = find(ssold >= smin*lf,1,'first');
ss    = ssold(indstart:end);                      
Nf    = length(ss)-1;                 % # of flagella elements
ssc   = (ss(2:end)+ss(1:(end-1)))/2;  % Centroids of flagella elements
                                      % [micron]
                                      
llf   =  ss(2:end)-ss(1:(end-1));     % Length of each flagella segment 
  


Y1  = curv2xy_quick(squeeze(curv),ssold,...
     pi/2,0,0);

% k1s = smooth(squeeze(kappasave(kk,2,2:end)),3);

xf1     = Y1(2:end,2);
yf1     = Y1(2:end,3);
zf1     = zeros(size(xf1)); 
% [xf1 yf1]

thf1    = Y1(2:end,1); 

Mf1 = flagella_mobility(Slend,rf,llf',Nf,xf1,yf1,zf1,thf1,ssc');

vxf1    = zeros(Nf,1) + U(1);
vyf1    = zeros(Nf,1) + U(2);
vzf1    = zeros(size(vxf1))+ U(3);


uu1 = [vxf1 vyf1 vzf1]';

Mf1

% f = Mf1\uu1(:);

% fq = reshape(f,3,Nf);
% Ftotal = sum(fq(1,:));

% plot(yf1,fq(1,:))

% [U_t,U_r]       = U_colloc(U,W,centroids,r/3);
