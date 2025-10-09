function d=dPipette(p,pv)
%% FUNCTION DPIPETTE
% Determines the distance of each point in pv to the closest point in p
%% INPUTS
%p      xyz coordinates of grid points                          [micron]
%pv     Points describing the contour of pipette the cell body  [micron]   
%% OUTPUTS
%d      Distance of each point in p to the closest? point in pv [micron]    

% Points
X     = p(:,1);
R     = sqrt(p(:,2).^2+p(:,3).^2);

pr    = [X R];
d     = dpoly(pr,pv);















