function [M] = flagella_mobility(Slend,rf,l,N,x,y,z,theta,SS) 

% Output:
%   - M  Assembled matrix  M*U=R of the linear system
% Input:
%   - l:           Vector of link length
%   - l_tot,Slend: Geometric parameter / total length / Slenderness
%   - N:           # of links
%   - x,y,theta:   Vector of link coordinate
% Last update:
% 20170310 by Daniel.
%============================== Define Parameters =========================================%
cst    = log(Slend^2);                                   
lf     = rf/Slend;

if min(l(:))<rf
    warning(['Flagellum elements smaller than flagellum radius\n'])
%           trial Da, 20170726
%     error('Flagellum elements smaller than flagellum radius')
end
%=========================== PreAllocate and declare variables ============================% 
K      = zeros(3*N,3*N);
L      = zeros(3*N,2);
H      = zeros(3*N,2);
one    = ones(1,N);
Id     = eye(N);

%====================== (I) COMPUTE ASSEMBLE MATRIX FOR FLAGELLUM =========================%
%=========================Compute Values for evaluation of Integrals=======================%
Xi     = x*one;
Yi     = y*one;
Zi     = z*one;

Li     = l*one;
cosus  = cos(theta)*one; 
sinus  = sin(theta)*one;
Si     = SS*one;
Lij    = Li'./Li;               % Matrix full of ones if all flag elements have the same length
Xij    = Xi-Xi';
Yij    = Yi-Yi';
Zij    = Zi-Zi';

% Add identity matrix to avoid division by zero (diagonals are zero)
Rij    = sqrt( Xij.^2 + Yij.^2 + Zij.^2)   + 10^20*Id;

Xij    = Xij./Rij; 
Yij    = Yij./Rij;
Zij    = Zij./Rij;

Sij    = abs(Si-Si')  + 10^20*Id;

% Select only non-diagonals
Mask   = ones(size(Rij))-eye(size(Rij));
%======================== action flagellum on flagellum =================================%
K(1:3:3*N,1:3:3*N) = (1 + Xij.^2)./Rij.*Mask; 
K(1:3:3*N,2:3:3*N) =   (Xij.*Yij)./Rij.*Mask;
K(2:3:3*N,1:3:3*N) =      K(1:3:3*N,2:3:3*N);
K(2:3:3*N,2:3:3*N) = (1 + Yij.^2)./Rij.*Mask;
K(3:3:3*N,3:3:3*N) = (1 + Zij.^2)./Rij.*Mask;

% K = 0*K;

L(1:3:3*N,1)       = sum((1 +   cosus.^2).*Lij./Sij.*Mask,2);
L(1:3:3*N,2)       = sum((  cosus.*sinus).*Lij./Sij.*Mask,2);
L(2:3:3*N,1)       = sum((  cosus.*sinus).*Lij./Sij.*Mask,2);
L(2:3:3*N,2)       = sum((1 +   sinus.^2).*Lij./Sij.*Mask,2);
L(3:3:3*N,3)       = sum((1             ).*Lij./Sij.*Mask,2);

% L = 0*L;
cst    = -log(4*(lf-SS).*SS./rf^2);

H(1:3:3*N,1)       = (-cst+1-(cst+3).*cosus(:,1).^2)         ./(l);
H(1:3:3*N,2)       = (      -(cst+3).*cosus(:,1).*sinus(:,1))./(l);
H(2:3:3*N,1)       = (      -(cst+3).*cosus(:,1).*sinus(:,1))./(l);
H(2:3:3*N,2)       = (-cst+1-(cst+3).*sinus(:,1).^2)         ./(l);
H(3:3:3*N,3)       = (-cst+1                       )         ./(l);


K(1:3:3*N,1:3:3*N) = 1/(8*pi)*(K(1:3:3*N,1:3:3*N) + diag(H(1:3:3*N,1)-L(1:3:3*N,1)));
K(1:3:3*N,2:3:3*N) = 1/(8*pi)*(K(1:3:3*N,2:3:3*N) + diag(H(1:3:3*N,2)-L(1:3:3*N,2))); 
K(2:3:3*N,1:3:3*N) = 1/(8*pi)*(K(2:3:3*N,1:3:3*N) + diag(H(2:3:3*N,1)-L(2:3:3*N,1)));
K(2:3:3*N,2:3:3*N) = 1/(8*pi)*(K(2:3:3*N,2:3:3*N) + diag(H(2:3:3*N,2)-L(2:3:3*N,2)));
K(3:3:3*N,3:3:3*N) = 1/(8*pi)*(K(3:3:3*N,3:3:3*N) + diag(H(3:3:3*N,3)-L(3:3:3*N,3)));

%======================Compute Assembled matrix: FLAGELLUM=================================%
M = K;  













