

home = fullfile('/home','sjoerd-buitjes','University',...
                                              'Master-Thesis','BEM','Boundary-Element-Method');

% codepath= fullfile(home,'code');

a            = 4.34; 
% a            =5;
% b            =4;       
b            = 3.73;       
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

%% Waveform of the flagella
waveformsfile        = fullfile(home,'datafiles','waveform','lib02_1_90_2019-06-28_1640.mat');
        
%% Filename to save the general results        
solutionfile       = fullfile(home,'datafiles','BEM-results','BEM-Flow.mat');

set(0,'defaulttextinterpreter','latex','DefaultAxesFontSize',16);
OTVfilename = [home filesep 'datafiles' filesep 'BEM-results' filesep 'FlowSpeed' filesep 'FlowSpeed_',...
                       '_',num2str(pt,'%02d'),'.mat'];


flowheadpath=fullfile(home,'datafiles','BEM-results','Mflow_head');

% add paths since code was built like this
addpath(fullfile(home,'code','initialisation'))
addpath(fullfile(home,'code','BEM'))
addpath(fullfile(home,'code','plotting'))
addpath(fullfile(home,'code','distmesh'))  
addpath(fullfile(home,'code','utils'))  