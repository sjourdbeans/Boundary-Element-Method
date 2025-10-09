%% DOC %%
% Update Log of Flow_Around_Chlamy... . m (FAC*.m)
% Date: 2014-06.      Created.     by Daniel Tam
% Date: 2017-07-18.   Organized.   Wei Da
%     Some organization is called for after several users has worked
%     with sults
%         User dependent. One may edit directly the FAC_004_SaveResults.m
%         file in the switchthis program. The organization of the code is as follows:
%      
%     001. Add paths to include functions and necessary files. 
%         As this code will be used on different computers, this part is
%         fully user dependent. Please modify the FAC_001_AddPath.m such
%         that your own filepaths are included in the switch structure
%     002. Set the operational mode of the code, load or creat necessary
%         parameters for the following simulation.
%         The structure of this part is not user dependent. In case that  
%         some values such as flag stiffness, flag radius need to be
%         changed, user should notify others or simply do it in one's local
%         version
%     003. Compute the flow field around a chlamy cell. 
%         FAC_003_CalcImgSequence_CORE.m
%         This part should not be modified as it directly affects the 
%         validity of the computation. Any change must be annouced to
%         all the users of this code and updating this DOC at the same time. 
%     004. Save simulation re structure.
%
% Date: 2017-11-13.   FAC_003_CORE updated by Daniel. 
%     1. Redundant calculation of Y1 and Y2 were deleted (it was Y1(2)s
%     that were used.
%     2. Flagella shape coordiantes were calculated now using
%     curv2xy_quick.m, instead of flagella_quick.m. The former is the one
%     how the clicked coordinates got into curvature in the first place.
% 
% Date: 2019-01-19.   FAC_003_CORE organized. 
%     1. Code organized, redundant part deleted.
%     2. Note that Phi (rate of work) lacks a kinematic viscosity 
%

%% Add paths  
% User dependent, can modify without noticing others.

addpath(fullfile(home,'code','initialisation'))
initialise_files


%% %%%%%%%%%%%%%%%%%%%%%%%%  Operational modes   %%%%%%%%%%%%%%%%%%%%%%%%%%
% User dependent, can modify without noticing others.

calc = 1;           % 0: Only show sequence of shapes
                    % 1: Do calculations
saveresults  = 1;   % Save BEM results?
compute_flow = 1;   % 0: Only solve integral equation
                    % 1: Solve integral equation on velocity on mesh
                    % 2: Solve integral equation and velocity on list of points
                    % 3: Both 1 and 2
flowshift   = 0;    % Phase shift for flow (positive=lead,negative=lag) [rad] 

makemovie   = 1;    % Make movie of flow field?
sepgridquiv = 1;    % Use reduced grid for quiver plot?

RemoveFlag1 = 0;    % Place the flag infinitely far away
RemoveFlag2 = 0;    % from the cell. 1 = Remove. Flag1: right

BackgroundFlow = 0; % Is there a background flow? 
                    % If yes, go setting it ~line220 

%% Before you load pipette be sure to run calccapstokes.m 

%% Set up Cell + Pipette system == Upload Cell mesh
load(meshfile);    % File with points on the cell mesh.

xh         = centroids(:,1);         % (x,y,z) coordinates of the ...
yh         = centroids(:,2);         % centroids of Cell mesh ...         
zh         = centroids(:,3);         % [micron]
ih         = find(abs(zh)<1);
Nh         = size(areas(:),1);       % # of elements representing the ...
                                     % cell + pipette head
                                     
theta      = linspace(0,2*pi,100);   % Angle     [rad]

% size parameters of the cell and the pipette a,b,d ...
% are stored in the pipettefile.mat

xhead      = a*cos(theta);           % x of head  [micron]
yhead      = b*sin(theta);           % y of head  [micron]
eta        = 0.9544e-3;              % Dynamic viscosity @22 degrees [Pa*s]

%% Set up Flagella & Upload shape file                   
load(waveformsfile); % File with waveforms

rf         = 0.2;                   % Radius of the flagellum, [micron]...           
                                     % 0.20+/-0.03, TEM Calib...
                                     % @20170601 

lf         = lf0;                    % Length of the flagellum  [micron]
Slend      = rf/lf;                  % Slenderness ratio        [-]
EI         = 0.9e-21;                % Bending stiffness        [N m^2]
kappasave(:,:,2:end) = kappasave(:,:,2:end)./lf;% Rescale curvature in ...
Rotmat = @(phi) [cos(phi) sin(phi); -sin(phi) cos(phi);];         % Rotation matrix
velx       = velx0.*lf;                         %               [micron/s]
vely       = vely0.*lf;                         %               [micron/s]
Nf         = size(kappasave,3)-2;     % # of flagella elements  [-]
ssold      = linspace(0,lf,Nf+1);     % Flagella segments       [micron]

smin       = 0.15;                    % Flagellum starts at...  [-]
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
                                      % [micron]

%%%%%%%%%%%%%%%%%%%%%%%%%
xbase = -Cell.dist_base;
ybase = 0  ;                          % y coord of flagellar base  [micron]
dtime = 1/fps;                        % Time step [s]

indh  = (1:3*Nh);                     % Index for head
indf1 = (1:3*Nf)+3*Nh;                % Index for flagellum 1 (RIGHT)
indf2 = (1:3*Nf)+3*(Nh+Nf);           % Index for flagellum 2 (LEFT)

M     = zeros(3*Nh+2*3*Nf,3*Nh+2*3*Nf); % Mobility matrix  [1e6 m/(N*s)]
M(1:3*Nh,1:3*Nh) = MATRIX;              % Include pre-calculated matrix

UU    = zeros(3*Nh+2*3*Nf,1);           % Velocities at each element's ...
                                        % boundary     [micron/s]
                                        
UU_max     = 100*lf;                    % Maximum velocity for plotting
                                        % [micron/s]

beginctr   = 1;                         % First frame to process
endctr     = size(kappasave,1);         % Last frame to process

nframes    = endctr-beginctr+1;         % Number of frames/time steps 
nframestot = size(kappasave,1);         % Total number of frames in data
BEMtime    = dtime.*(0:1:nframestot-1); % Time vector              [s]

%% Set up a cartesian Grid in the bulk on which to compute the flow.
if(compute_flow == 1) || (compute_flow == 3)
    Nxg       = 120+1;                  % Number of nodes in x axis[-]
    Nyg       = 80+1;                   % Number of nodes in y axis[-]
    xmin = -30; xmax = 30; 
    ymin = -20; ymax = 20;              % Limits for velocity grid
    [xg,yg]   = ndgrid(linspace(xmin,xmax,Nxg),...
                linspace(ymin,ymax,Nyg));   % Grid xy    [micron]
    zg        = zeros(size(xg));            % Grid z     [micron]
    xg = xg(:);  
    yg = yg(:); 
    zg = zg(:);                         % Make matrices to column vectors
    Ng = size(xg,1);                    % Number of grid points
    if sepgridquiv == 1
        [xgq,ygq] = ndgrid(linspace(xmin,xmax,(Nxg-1)/4+1),...
                           linspace(ymin,ymax,(Nyg-1)/4+1));           
                                        % Grid for quiver [micron]
        zgq = zeros(size(xgq));         % [micron]
        xgq = xgq(:);
        ygq = ygq(:); 
        zgq = zgq(:);      
        Ngq        = size(xgq,1);       % Number of grid points
        indf2q = zeros(size(xgq));
        for ii=1:length(xgq)            % Find indices of full grid 
            indx = find(xg == xgq(ii)); % corresponding to compact grid
            indy = find(yg == ygq(ii));
            [~,ind] = ismember(indx,indy);
            indf2q(ii) = indx(find(ind~=0,1,'first'));
        end
    end
end

%% Set up location(s) to extract flow speed time series
if compute_flow >= 2
    % Comparing OTV, separate bead positions
    % [xgb,ygb]  = BeadCoordsFromFile(beadcoords_pth,pt_list,experiment);
    % OR: An axial slice:
    xgb = (18:2:125)*-1; ygb = zeros(size(xgb));        % I commented out the Bead coordinates line - Sjoerd
    % OR: A lateral slice:
    % ygb = [18:2:125]*-1; xgb = zeros(size(xgb)); 
    
    zgb = zeros(size(ygb));             % zero height
    Nb = length(xgb);
end

%% Compute flow matrix for the head
if compute_flow ~= 0
    % Grid points closer than XXX micron to the pipette+cell body are set
    % to 0 flow speed
    % indg are the indices of those points
    indg     = find(dPipette([xg yg zg],pv) <= 0.1);
    % indg =[];  
    if ~exist('xmax','var')
        xmax = 1; ymax = 1;
        Nxg  = 1; Nyg  = 1; 
    end    
    % [~,ParentDir,~]   = fileparts(fileparts(pwd));      %probably should return name
    % Naming can be more specific to make sure the correct file is used
    flowheadfile      = ['M_Flow_head_spheroid_2',...
                        sprintf('_%d_%d_%dx_%dy.mat',...
                         Nxg,Nyg,xmax,ymax)];


    flowheadfullpath  = [flowheadpath filesep flowheadfile];  % "\" replaced with filesep
    MFlowHeadExist    = exist(flowheadfullpath,'file');
    if (compute_flow == 1) || (compute_flow == 3)
        if MFlowHeadExist
            load(flowheadfullpath)
        else
            [M_Flow_head,~,~,~,~] = FlowStokes(panels,...
                                    [xg(:) yg(:) zg(:)],...
                                    normals,param);
            save(flowheadfullpath,'M_Flow_head')
        end
    end    
    if compute_flow >= 2
        [M_Flow_headb,~,~,~,~] = FlowStokes(panels,...
                                 [xgb(:) ygb(:) zgb(:)],...
                                 normals,param);
        [Uflowb,Vflowb,Wflowb] = deal(zeros(nframes,length(xgb)));
    end
end

%% Set up External flow
%{
velocity of a translating pipette in stationary medium, 
this corresponds to MINUS the velocity of the flow on a 
stationary pipette.
U is the translation velocity of pipette     [micron/s]
W is the rotation velocity of pipette        [rad/s]
%}
if BackgroundFlow == 0
    [U,W] = deal(zeros(3,nframes));
    % W(3,:)=ones(1,nframes);
else 
    W = zeros(3,nframes);
    U = setup_piezo_flow_Da('00NoFlow',49.2,800,nframes,fps);    
    % inputs are: flow direction, frequency, amplitude (NOT pp-ampl)
    U(1) = 0; U(2) = 0; U(3) = 0;     % historical codes,
    W(1) = 0; W(2) = 0; W(3) = 0;     % may be necessary           
end

%% Calculation
% User independent, MODIFICATION MUST BE ANNOUCED.
Flow_Around_Chlamy_003_ComputeForceAndFlow_CORE

%% Save results. 
% User dependent, can modify without noticing others.
Flow_Around_Chlamy_004_SaveResults
