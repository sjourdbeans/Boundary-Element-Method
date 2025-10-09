%% Doc
% This file is user dependent. 
% Used as the last part to save movies and simulation results.
%


%% End of the whole routine
% names sorted in alphabetical ascending order

if compute_flow >= 2
    Uflowb = - Uflowb;   Vflowb = - Vflowb;	
    % this is for comparing with optical trap data.
    save(OTVfilename,'Uflowb','Vflowb','xgb','ygb',...
        'fps','BEMtime','fx1','fx2','phi1','phi2')
    % recover U,V flow orientation
    Uflowb = - Uflowb;   Vflowb = - Vflowb;
end

%% SAVE SIMULATION RESULTS
if saveresults
    clear areas centroids normals panels
    clear COLN_R COLN_S LINE_R LINE_S phi RHS
    clear M MATRIX MATRIX_h1 MATRIX_h2 Mf1 Mf12 Mf1h Mf2 Mf21 Mf2h
    save(solutionfile,'-regexp','^(?!(M_Flow_head)$).');
    % M_Flow_head is of 1GB size. It flooded the hard disk and it
    % is identical for cases from the same scenario.
end

%% MAKING MOVIE
if makemovie ~= 0
    moviefilename = [OTVfilename(1:end-4),'.avi'];
    v = VideoWriter(moviefilename,'Motion JPEG AVI');
    v.FrameRate = 5;
    open(v)
    writeVideo(v,F(2:length(F)))
    close(v)
end



