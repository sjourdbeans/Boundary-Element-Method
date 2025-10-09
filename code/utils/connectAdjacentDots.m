% Doc
% construct sorted x,y coordinates based on distance from on to another
function [x_connected,y_connected] = connectAdjacentDots(x,y)
    Nx = length(x); Ny = length(y);
    if Nx ~= Ny
        error('Input coordinates must have the same size')
    else 
        N = Nx;
    end    
    
    x_connected = zeros(N,1);
    y_connected = zeros(N,1);
    x_connected(1) = x(1); 
    y_connected(1) = y(1);
    x(1) = [];
    y(1) = [];
    
    for i = 2:N
        x_prev =  x_connected(i-1); 
        y_prev =  y_connected(i-1);
        
        N_restElements = length(x);
        d_toTheRest = zeros(N_restElements,1);
        for j = 1:N_restElements
            d_toTheRest(j) = distance([x_prev,y_prev],[x(j),y(j)]);
        end
        [~,idx_closest] = min(d_toTheRest);
        x_next = x(idx_closest);
        y_next = y(idx_closest);
        x(idx_closest) = [];
        y(idx_closest) = [];
        
        x_connected(i) = x_next;
        y_connected(i) = y_next;
    end
    
    function d = distance(coord1,coord2)
        d =sqrt(sum((coord1-coord2).^2));
    end
end