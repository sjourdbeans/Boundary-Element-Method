function [M] = flagella_interaction(xf,yf,zf,xh,yh,zh,Nf,Nh,rad) 
%% FUNCTION FLAGELLA_INTERACTION
% Computes the mobility matrix of interaction between two features
%% INPUTS
%xf         x coordinate of first feature               [micron]
%yf         y coordinate of first feature               [micron]
%zf         z coordinate of first feature               [micron]
%xh         x coordinate of second feature              [micron]
%yh         y coordinate of second feature              [micron]
%zh         z coordinate of second feature              [micron]
%Nf         Number of elements for first feature        [-]
%Nh         Number of elements for second feature       [-]
%rad        Radius of flagellum                         [micron]
%% OUTPUTS
%M          Assembled matrix  M*U=R of the linear system

%=========================== PreAllocate and declare variables ============================% 
M      = zeros(3*Nh,3*Nf);
onef   = ones(1,Nf);
oneh   = ones(Nh,1);

%====================== (I) COMPUTE ASSEMBLE MATRIX FOR FLAGELLUM =========================%
%=========================Compute Values for evaluation of Integrals=======================%
Xi     = xh(:)*onef;
Yi     = yh(:)*onef;
Zi     = zh(:)*onef;

Xj     = oneh*xf(:)';
Yj     = oneh*yf(:)';
Zj     = oneh*zf(:)';

Xij    = Xi-Xj;
Yij    = Yi-Yj;
Zij    = Zi-Zj;

Rij    = sqrt( Xij.^2 + Yij.^2 + Zij.^2)+0.5*rad;

Xij    = Xij./Rij; 
Yij    = Yij./Rij;
Zij    = Zij./Rij;
%======================== action flagellum on flagellum =================================%
M(1:3:end,1:3:end) =  (1 + Xij.^2)./Rij + rad^2/2*(1 -3*   Xij.^2)./Rij.^3; 
M(1:3:end,2:3:end) =    (Xij.*Yij)./Rij + rad^2/2*(  -3* Xij.*Yij)./Rij.^3;
M(1:3:end,3:3:end) =    (Xij.*Zij)./Rij + rad^2/2*(  -3* Xij.*Zij)./Rij.^3;
M(2:3:end,1:3:end) =                                    M(1:3:end,2:3:end);
M(2:3:end,2:3:end) =  (1 + Yij.^2)./Rij + rad^2/2*(1 -3*   Yij.^2)./Rij.^3;
M(2:3:end,3:3:end) =    (Yij.*Zij)./Rij + rad^2/2*(  -3* Yij.*Zij)./Rij.^3;
M(3:3:end,1:3:end) =                                    M(1:3:end,3:3:end);
M(3:3:end,2:3:end) =                                    M(2:3:end,3:3:end);
M(3:3:end,3:3:end) =  (1 + Zij.^2)./Rij + rad^2/2*(1 -3*   Zij.^2)./Rij.^3;

%======================Compute Assembled matrix: FLAGELLUM=================================%
M = 1/(8*pi)*M;  
