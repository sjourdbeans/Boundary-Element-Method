function [U_t,U_r] = U_colloc(U,W,centroid,r)
%% FUNCTION U_COLLOC
% Calculates the translational and rotational velocity of the background
% flow at each point to form the big U vector (RHS).
%% INPUTS
%U          Translational velocity of external flow     [micron/s]
%W      	Rotational velocity of external flow        [rad/s]
%centroid   xyz coordinates of centroids of cell mesh   [micron]
%r          Number of rows of mobility matrix           [-]
%% OUTPUTS
%U_t        Translational velocity                      [micron/s]
%U_r        Rotational velocity                         [micron/s]

U_t = zeros(3*r,1);
U_r = zeros(3*r,1);
for i=1:r
    U_t(3*(i-1)+1,1) = U(1);    
    U_t(3*(i-1)+2,1) = U(2);
    U_t(3*(i-1)+3,1) = U(3);
    
    U_r(3*(i-1)+1,1) = W(2)*centroid(i,3) -  W(3)*centroid(i,2);
    U_r(3*(i-1)+2,1) =-W(1)*centroid(i,3) +  W(3)*centroid(i,1); 
    U_r(3*(i-1)+3,1) = W(1)*centroid(i,2) -  W(2)*centroid(i,1); 
end