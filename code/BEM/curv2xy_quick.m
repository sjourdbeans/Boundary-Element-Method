function [Ytot] = curv2xy_quick(curv,ssc,theta0,x0,y0)
%% FUNCTION CURV2XY_QUICK 
%Generates xy coordinates from a curvature profile
%% INPUTS
%curv       Curvature profile at points ssc
%ssc        Grid of points where output is desired
%theta0     Starting tangent angle [rad]
%x0         Starting x [px]
%y0         Starting y [px]
%% OUTPUTS
%Ytot       Matrix with tangent angle, x and y vectors
    ds          = ssc(2)-ssc(1);
    Ytot(1,1)   = theta0;
    Ytot(1,2)   = x0;
    Ytot(1,3)   = y0;
    
    intpfac     = 10; %Interpolate each step n times 
    dsintp      = ds/intpfac;
    for ii=2:length(ssc)
        Ytottemp = Ytot(ii-1,:);
        for jj=1:intpfac
            %Take curvature value midway between interpolation points
            curvintp = curv(ii-1) + (2*jj-1)/(2*intpfac)*(curv(ii)-curv(ii-1));
            Ytottemp(1) = Ytottemp(1) + dsintp*curvintp;
            Ytottemp(2) = Ytottemp(2) + dsintp*cos(Ytottemp(1));
            Ytottemp(3) = Ytottemp(3) + dsintp*sin(Ytottemp(1));
        end
        Ytot(ii,1) = Ytottemp(1);
        Ytot(ii,2) = Ytottemp(2);
        Ytot(ii,3) = Ytottemp(3);
    end
end

