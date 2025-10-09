function [area,centroid,Z,FD,LS,LR,S] = calcpStokes(panel,evalpnts,p)
%% FUNCTION CALCPSTOKES
% Matlab version of calcp, returns potential at evaluation point due
% to unit monopole and unit dipole uniformly distributed on a panel.
% Follows a left-hand rule (Clockwise ordered points has normal
% pointing up).
%% INPUTS
%panel      Vectors of panel vertices in rows of x,y,z (3 or 4 rows supported)
%evalpnts   Matrix of evaluation points, rows of x,y,z coordinates
%p          param struct defined in calccapStokes.m (center line coordinates)
%% OUTPUTS
%area       Panel area
%centroid   Panel centroid.
%Z          Panel normal
%FD         Force dipole?
%LS         Line Stokeslet?
%LR         Line Rotlet?
%S          ????

% fess = the derivative of the monopole potential at evalpnt along direction
% fess = the derivative of the dipole potential at evalpnt along direction
% fss = the vector of potentials due to a monopole
% fds = the vector of potentials due to a panel normal dipole distribution 

% First check the input.
[verts, betterbethree] = size(panel);

if betterbethree ~= 3
  error('wrong panel format: should be rows of x,y,z vectors!')
end

if (verts > 4) || (verts < 2) 
  error('wrong panel format: panel can only have 3 or 4 vertices!')
end

if(nargin > 1) 
  [numevals, betterbethree] = size(evalpnts);
  if betterbethree ~= 3
    error('wrong evaluation point format: should be rows of x,y,z vectors!')
  end
else
  numevals = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%----------------------------I PANEL SETUP--------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Length of each side and the panel area.
for i=1:verts
  if(i < verts)
    side(i,:) = panel(i+1,:) - panel(i,:);
  else 
    side(i,:) = panel(1,:) - panel(i,:);
  end
  edgeLength(i) = norm(side(i,:));
end

% Calculate the panel coordinate system.
X = panel(3,:) - panel(1,:);
diagLength = norm(X);
if(verts == 3) 
  Y = panel(2,:) - panel(1,:); 
else 
  Y = panel(2,:) - panel(4,:); 
end

% Z-axis is normal to two diags.
Z = cross(X, Y);
area = 0.5 * norm(Z);

% Normalize panel axise s. 
coord(3,:) = Z / norm(Z);
coord(1,:) = X / norm(X);
X = coord(1,:);
Z = coord(3,:);
coord(2,:) = cross(Z, X);
Y = coord(2,:);


% Determine the centroid.
vertex1 = panel(2,:) - panel(1,:);
if(verts == 4)
  vertex3 = panel(4,:) - panel(1,:);
else 
  vertex3 = panel(3,:) - panel(1,:);
end

% Project into the panel axes.
y1 = sum(vertex1 .* Y);
y3 = sum(vertex3 .* Y);
x1 = sum(vertex1 .* X);
x3 = sum(vertex3 .* X);
yc = (y1 + y3)/3.0;
xc = (x1 + x3)/3.0;

% Compute the centroid.
centroid = panel(1,:) + xc * X + yc * Y;

% Put the corners in the newly defined coordinate system.  
for i=1:verts
  npanel(i,:) = (coord * (panel(i,:) - centroid).').';
end

% Check that panel is in the x-y panel. 
for i=1:verts
    if(abs(npanel(i,3)) > (1.0e-8 * diagLength))
        npanel
        error('Coordinate transform failure!!')
    end;
end;

% Compute quadrature points and weights for the panel (triang quad)
[Xq,Yq,Wx,Wy]=triquad(2,npanel(:,1:2));
Zq           = zeros(size(Xq));


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% II Matrix associated with double layer potential%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% loop through the evaluation (collocation) points!!
% [here we compute the contribution of each triangular element to the integral equation at each collocation point]

FD = [];

for evalindex = 1:numevals
    D     = zeros(3,3);         %Double layer potential

    Col     = coord * evalpnts(evalindex,:)';  % Collocation point
    Int     = coord * centroid';               % Centroid of the current surface element for the integral

    % xx, yy, zz contain 4 quad points
    xx    =  Col(1)-(Int(1)+Xq);  xx2 = xx.^2; % this is coord of collocation point - coord integration variable in rotated frame of reference
    yy    =  Col(2)-(Int(2)+Yq);  yy2 = yy.^2;  
    zz    =  Col(3)- Int(3);      zz2 = zz.^2;  

    r2    =  xx2 + yy2 + zz2;
    r     =  sqrt(r2);
    r5    =  r.^5;

    % Only calculate the double layer potential on the surface normal. i.e. T_ij3
    rr1   = zz*xx2 ./ r5;
    rr2   = zz*yy2 ./ r5;
    rr3   = zz*xx.*yy ./ r5;
    rr4   = zz*xx.*zz ./ r5;
    rr5   = zz*yy.*zz ./ r5;
    rr6   = zz*zz2 ./ r5;

    % Gauss quadrature
    D(1,1)= Wx'*rr1*Wy;
    D(2,2)= Wx'*rr2*Wy;
    D(1,2)= Wx'*rr3*Wy;
    D(1,3)= Wx'*rr4*Wy;
    D(2,3)= Wx'*rr5*Wy;
    D(3,3)= Wx'*rr6*Wy;  

    % Symmetric matrix
    D     = D + triu(D,1)';   
    %%%%%%%%%%%%% Line Distribution of Stokeslet and Rotlet %%%%%%%%%%%%%%%%%%%
    S     = zeros(3,3);  
    % Calculate X coordinate on the centerline in regular coordinate frame (NOT PANEL FRAME)
    % Geometry is axisymmetric (??) Ask Daniel

    % X coordinates of the quadrature points from the center of the line distribution (??).
    R     = p.e*(centroid(1)+Xq*coord(1,1)+Yq*coord(2,1)-p.XG) + p.XG;

    Rx    = R*coord(1,1);
    Ry    = R*coord(2,1);
    Rz    = R*coord(3,1);

    Px    = Col(1)-Rx; Px2 = Px.^2;     %(x_i - x_0i)       [micron]
    Py    = Col(2)-Ry; Py2 = Py.^2;     %(x_j - x_0j)       [micron]
    Pz    = Col(3)-Rz; Pz2 = Pz.^2;     %(x_k - x_0k)       [micron]
    PP    = sqrt(Px2 + Py2 + Pz2);      % r=|x-x0|          [micron]
    PP3   = PP.^3;                      % r^3               [micron^3]

    Qx    = Int(1)+Xq-Rx;
    Qy    = Int(2)+Yq-Ry;
    Qz    = Int(3)+Zq-Rz;

    trace = Px.*Qx+Py.*Qy+Pz.*Qz;

    s11   = 1./PP + Px2./PP3    + (trace - Qx.*Px)./PP3;
    s12   =         Px.*Py./PP3 + (      - Qx.*Py)./PP3;
    s13   =         Px.*Pz./PP3 + (      - Qx.*Pz)./PP3;

    s21   =         Py.*Px./PP3 + (      - Qy.*Px)./PP3;
    s22   = 1./PP + Py2./PP3    + (trace - Qy.*Py)./PP3;
    s23   =         Py.*Pz./PP3 + (      - Qy.*Pz)./PP3;

    s31   =         Pz.*Px./PP3 + (      - Qz.*Px)./PP3;
    s32   =         Pz.*Py./PP3 + (      - Qz.*Py)./PP3;
    s33   = 1./PP + Pz2./PP3    + (trace - Qz.*Pz)./PP3;

    S(1,1)= Wx'*s11*Wy ;
    S(1,2)= Wx'*s12*Wy ;
    S(1,3)= Wx'*s13*Wy ;

    S(2,1)= Wx'*s21*Wy ;
    S(2,2)= Wx'*s22*Wy ;
    S(2,3)= Wx'*s23*Wy ;

    S(3,1)= Wx'*s31*Wy ;
    S(3,2)= Wx'*s32*Wy ;
    S(3,3)= Wx'*s33*Wy ;

    D = coord' * ( 3/(4*pi)*D + 1/(8*pi)*S ) * coord ; % Back in the initial coordinate system

    FD = [FD ; D];   
end

S   = Wx'*Wy;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% III Matrix associated point stokestlet and rotlet in the o%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% quadrature points Coordinates in local coordinate system 
cent_pt   = (coord * centroid.').'; % (coord integration variable in local coord syst)
xx    =  cent_pt(1)+Xq;  
yy    =  cent_pt(2)+Yq;  
zz    =  cent_pt(3)+zeros(size(Xq));

% 1./ Start with the lines (required matrices over integration elements)
% Stokeslet
Surf= Wx'*ones(size(Xq))*Wy;
LS  = Surf*eye(3);

% Rotlet
LR      = zeros(3,3);
LR(1,2) =-Wx'*zz*Wy;
LR(1,3) = Wx'*yy*Wy;
LR(2,3) =-Wx'*xx*Wy;
LR(2,1) =-LR(1,2);
LR(3,1) =-LR(1,3);
LR(3,2) =-LR(2,3);

LR      = coord'*LR*coord;      % Back in the initial coordinate system
