

home = fullfile('/home','sjoerd-buitjes','University',...
                                              'Master-Thesis','BEM','Boundary-Element-Method');

% codepath= fullfile(home,'code');

% a            = 4.34; 
a            =1;
b            =1;       
% b            = 3.73;       
d            = 2.87;        
beadsize     = 2.78;   

flag_length = 10.88; 
lf0         = flag_length;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% supfld = [uni_name,'/'];    % name of the experiment
% cd(supfld)

pt = 96;            % name of the case (position) to analyze

% casefld_fullpath   = [topfld,supfld,num2str(pt,'%02d'),'/'];
% cd(casefld_fullpath)


%% Pipette shape
% meshfile        = [home,'/datafiles/mesh/pipette/pipette_25_400_300.mat'];
% meshfile       = fullfile(home,'datafiles','mesh','pipette','pipette_paper.mat');
meshfile       = fullfile(home,'datafiles','mesh','spheroid','mesh_panels_pv.mat');
% meshfile       = fullfile(home,'datafiles','mesh','spheroid','spheroid_00.mat');
% meshfile       = fullfile(home,'datafiles','mesh','spheroid','spheroid_01.mat');
% meshfile       = fullfile(home,'datafiles','mesh','spheroid','spheroid_05.mat');
% meshfile       = fullfile(home,'datafiles','mesh','spheroid','spheroid_mesh.mat');
% meshfile       = fullfile(home,'datafiles','mesh','sphere','sphere_mesh.mat');

refinement = {
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=1.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=6.500000e-01.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=5.000000e-01.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=4.000000e-01.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=2.500000e-01.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=2.000000e-01.mat')
    fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=1.000000e-01.mat')
    % fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=8.000000e-02.mat')
    % fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=5.000000e-02.mat')

};

spheroids = {
    fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.05.mat')
    fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.06.mat')
    fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.07.mat')
    fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.08.mat')
    fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.09.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.1.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.15.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.2.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.25.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.3.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.35.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.4.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.45.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.5.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.55.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.6.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.65.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.7.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.75.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.8.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.85.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.9.mat')
    % fullfile(home,'datafiles','mesh','spheroid-variation','spheroid_mesh_b=0.95.mat')

    % fullfile(home,'datafiles','mesh','sphere_refinement','sphere_mesh_h=5.000000e-02.mat')

};




%% Waveform of the flagella
waveformsfile        = fullfile(home,'datafiles','waveform','lib02_1_90_2019-06-28_1640.mat');
        
%% Filename to save the general results        
solutionfile       = fullfile(home,'datafiles','BEM-results','BEM-Flow.mat');

set(0,'defaulttextinterpreter','latex','DefaultAxesFontSize',16);
OTVfilename = [home filesep 'datafiles' filesep 'BEM-results' filesep 'FlowSpeed' filesep 'FlowSpeed_005',...
                       '_',num2str(pt,'%02d'),'.mat'];


flowheadpath=fullfile(home,'datafiles','BEM-results','Mflow_head');

% add paths since code was built like this
addpath(fullfile(home,'code','initialisation'))
addpath(fullfile(home,'code','BEM'))
addpath(fullfile(home,'code','plotting'))
addpath(fullfile(home,'code','distmesh'))  
addpath(fullfile(home,'code','utils'))  