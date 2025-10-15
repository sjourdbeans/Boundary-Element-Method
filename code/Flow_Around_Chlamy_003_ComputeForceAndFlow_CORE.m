%% Doc
%{
This code computes:
   1. The distribution of singularities over the mesh,
   2. Interaction matrix between the two flagella, and 
      pipette+cell body)
   3. The forces on flagella
   4. Rate of work
   5. Flagellar bending energies
   6. Flow field based upon the force distribution, if asked. 
   7. Time series of the flow vectors from one point or an array of 
      locations(bead position/positions), if asked.
 
 Inputs are generally the flagellar shapes, pipette+cell body mesh, and
 the flow field wherein the cell beats.

 To bored user: log file of code updates at the end 
%}

%% Pre-allocation
% Frames for making movie
clearvars F 
F(length(BEMtime)) = struct('cdata',[],'colormap',[]); 

% Energy
Ebend              = zeros(nframestot,3); % Bending energy
Pbend              = zeros(nframestot,2); % 

% Force vector (x&y) on every segment for flagella 1&2
[fx1,fy1,fx2,fy2]  = deal(zeros(nframestot,Nf));        

% Total drag force vector (x&y), for flagella 1&2
[D1,D2]            = deal(zeros(nframestot,2));       

%phi is the rate of work            [1e-18 W]
[phi1,phi2,Dtot]   = deal(zeros(nframestot,1));   
[r,c]              = size(MATRIX);              %Rows/columns


%%%%%%%%%%%%%%%%%%%TEST%%%%%%%%%%%%%%%%%
Cell.phi_body = pi/2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Computation
for kk = beginctr:endctr
    fprintf('Frame %d out of %d \n',kk-beginctr+1,nframes);
    %% Get flagellar shapes from stored variable
    % Compute the position vector of the flagellum
    % 1 = right, 2 = left in this program, whereas 1 = left, 2 = right in
    % detection...
    Y1  = curv2xy_quick(-squeeze(kappasave(kk,2,2:end)),ssold,...
        pi-(Cell.thetar-Cell.phi_body)-squeeze(kappasave(kk,2,1)),xbase,ybase);
    Y2  = curv2xy_quick(squeeze(kappasave(kk,1,2:end)),ssold,...
        pi-(Cell.thetal-Cell.phi_body)+squeeze(kappasave(kk,1,1)),xbase,ybase);
    k1s = smooth(squeeze(kappasave(kk,2,2:end)),3);
    k2s = smooth(squeeze(kappasave(kk,1,2:end)),3);

    % Cut out the proximal fraction (smin) of flagellar shapes 
    % to ensure flagella-body separation
    Y1 = Y1(indstart:end,:);
    Y2 = Y2(indstart:end,:);

    %x coordinate of flagellum      [micron]
    xf1     = Y1(2:end,2) + RemoveFlag1*lf*10^9;   
    xf2     = Y2(2:end,2) + RemoveFlag2*lf*10^9;             

    %y coordinate of flagellum      [micron]
    yf1     = Y1(2:end,3) + RemoveFlag1*lf*10^9;     
    yf2     = Y2(2:end,3) + RemoveFlag2*lf*10^9;        

    %z coordinate of flagellum      [micron]
    zf1     = zeros(size(xf1));   
    zf2     = zeros(size(xf2));     

    % save the coordinate of flagellum over time      [micron]
    flagx(kk,1,:) = xf1;
    flagx(kk,2,:) = xf2;
    flagy(kk,1,:) = yf1;
    flagy(kk,2,:) = yf2;
    flagz(kk,1,:) = zf1;
    flagz(kk,2,:) = zf2;

    % Tangent angle of flagellum     [rad]
    thf1    = Y1(2:end,1);     
    thf2    = Y2(2:end,1);        

    %% Calculate bending energy
    Ebend(kk,1) = 1/2*EI*lf/Nf*sum(k1s.^2)*1e6;     %Bending energy    [J]
    Ebend(kk,2) = 1/2*EI*lf/Nf*sum(k2s.^2)*1e6;     %Bending energy    [J]
    Ebend(kk,3) = Ebend(kk,1)+Ebend(kk,2);
       
    %% Rotate and scale velocity vectors
    for lr=1:2
        if lr == 1 %Left flagellum in storage, right in this program
            phi = pi+(Cell.thetal-Cell.phi_body);
        else %Right flagellum in storage, left in this program
            phi = pi-(Cell.thetar-Cell.phi_body);
        end
        vrot = [squeeze(velx(kk,lr,:))';squeeze(vely(kk,lr,:))';];
        vrot = Rotmat(phi)*vrot; 
        velx(kk,lr,:) = vrot(1,:); %#ok<*SAGROW>
        if lr == 1
            vely(kk,lr,:) = vrot(2,:);
        else
            vely(kk,lr,:) = -vrot(2,:);
        end
    end
    
    %% Include background flow velocity here 
    vxf1    = squeeze(velx(kk,2,indstart+1:end)) + U(1,kk);
    vxf2    = squeeze(velx(kk,1,indstart+1:end)) + U(1,kk);
    vyf1    = squeeze(vely(kk,2,indstart+1:end)) + U(2,kk);
    vyf2    = squeeze(vely(kk,1,indstart+1:end)) + U(2,kk);
    vzf1    = zeros(size(vxf1))+ U(3,kk);
    vzf2    = zeros(size(vxf1))+ U(3,kk);
    
    %% Either only check flagellar shapes 
    if calc == 0
        % Check waveforms and velocity vectors
        figure(1),clf,hold on;
        plot(xhead,yhead,'k','LineWidth',1); hold on;
        plot(Y1(2:end,2),Y1(2:end,3),'g','LineWidth',0.8)
        plot(Y2(2:end,2),Y2(2:end,3),'r','LineWidth',0.8)
        if compute_flow >= 2
           plot(xgb,ygb,'bo') 
        end
        xlabel('x [$\mathrm{\mu}$m]'),ylabel('y [$\mathrm{\mu}$m]')
        grid on,axis equal,axis([-15 10 -10 10]);
        set(gca, 'xtick', [-10 -5 0 5 10], 'ytick', [-10 -5 0 5 10])
        pause  
    end
    
    %% Or Compute flow and force for all
    if calc
        %% Compute interaction matrices
        % Assemble RHS

        % Translation/rotational velocity vectors    [micron/s]
        [U_t,U_r]       = U_colloc(U(:,kk),W(:,kk),centroids,r/3);

        % Right hand side (velocity)                 [micron/s]
        RHS             = (U_t+U_r);                                    

        % Put together the Mobility matrix
        % Headf
        M(indh,indh) = MATRIX;   

        % Flagella 
        Mf1 = flagella_mobility(Slend,rf,llf',Nf,xf1,yf1,zf1,thf1,ssc');
        Mf2 = flagella_mobility(Slend,rf,llf',Nf,xf2,yf2,zf2,thf2,ssc');
        M(indf1,indf1) = Mf1;
        M(indf2,indf2) = Mf2;

        % Flagella interaction
        Mf12 = flagella_interaction(xf1,yf1,zf1,xf2,yf2,zf2,Nf,Nf,rf);
        Mf21 = flagella_interaction(xf2,yf2,zf2,xf1,yf1,zf1,Nf,Nf,rf); 
        Mf1h = flagella_interaction(xf1,yf1,zf1,xh,yh,zh,Nf,Nh,rf); 
        Mf2h = flagella_interaction(xf2,yf2,zf2,xh,yh,zh,Nf,Nh,rf); 
        M(indf2,indf1) = Mf12;
        M(indh,indf1)  = Mf1h;
        M(indf1,indf2) = Mf21;
        M(indh,indf2)  = Mf2h;

        % Head Interaction
        [MATRIX_h1,~,~,~,~] = FlowStokes(panels,[xf1 yf1 zf1],normals,param);
        [MATRIX_h2,~,~,~,~] = FlowStokes(panels,[xf2 yf2 zf2],normals,param);
        M(indf1,indh) = MATRIX_h1;
        M(indf2,indh) = MATRIX_h2;

        % Put together the Velocity matrix : 
        % VELOCITY AT SURFACE OF HEAD AND PIPETTE REMAINS ZERO!!!
        uu1 = [vxf1 vyf1 vzf1]';       
        uu2 = [vxf2 vyf2 vzf2]';
        UU(indf1)  = uu1(:); 
        UU(indf2)  = uu2(:); 
        UU(indh)   = RHS(:);

        %% Compute force and rate of work
        phi        = M\UU;
        
        %Point forces
        fh         = phi(indh);         % for the head         [1e-12 N]
        f1         = phi(indf1);        % for the right flag   [1e-12 N]
        f2         = phi(indf2);        % for the left flag    [1e-12 N]

        % Compute Total rate of work & Drag force on each flagellum
        fx1(kk,:)  = f1(1:3:end); 
        fy1(kk,:)  = f1(2:3:end);
        fx2(kk,:)  = f2(1:3:end); 
        fy2(kk,:)  = f2(2:3:end);

        phi1(kk) = fx1(kk,:)*(vxf1-U(1,kk)) + ... % Rate of work, 
                   fy1(kk,:)*(vyf1-U(2,kk));      % right flag [1e-18 W]
        phi2(kk) = fx2(kk,:)*(vxf2-U(1,kk)) + ...
                   fy2(kk,:)*(vyf2-U(2,kk));      % left flag  [1e-18 W]
               
        D1(kk,:) = [sum(fx1(kk,:)) ; ...          % Drag on the
                    sum(fy1(kk,:))      ];        % right flag [1e-18 N] 
        D2(kk,:) = [sum(fx2(kk,:)) ; ...
                    sum(fy2(kk,:))      ];        % left flag  [1e-18 N]
                                                  % UNIT COULD BE WRONG  
        Dtot(kk) = D1(1)*U(1) + D1(2)*U(2);       % Total rate of work
        
        %% Compute flow on the cartesian grid: [xg,yg,zg]
        if (compute_flow == 1) || (compute_flow == 3)
            %% compute flow field
            M_Flow_f1  = flagella_interaction(...% Flow matrix
                             xf1,yf1,zf1,...     % for the right flag 
                             xg,yg,zg,...
                             Nf,Ng,rf);   
            M_Flow_f2  = flagella_interaction(...
                             xf2,yf2,zf2,...     % for the left flag
                             xg,yg,zg,...
                             Nf,Ng,rf);   
            UF         = M_Flow_head*fh(:)  + ...% Velocity field  
                         M_Flow_f1  *f1(:)  + ...% [micron/s]
                         M_Flow_f2  *f2(:) ;     

            % Subtract background flow from the result to correct 
            % for boundary condition
            u_flow  = UF(1:3:end)-U(1,kk);   % x velocity [micron/s]
            v_flow  = UF(2:3:end)-U(2,kk);   % y velocity [micron/s]
            w_flow  = UF(3:3:end)-U(3,kk);   % z velocity [micron/s]
            u_flow(indg)= 0;                 % Zero boundary velocity 
            v_flow(indg)= 0;
            w_flow(indg)= 0;

            %% plot flow field and the cell for each frame
            figure(1);  clf;  hold on;
            title(sprintf('t = %4.3f ms',1000*BEMtime(kk)))
            
            UU_flow       = sqrt(u_flow.^2+v_flow.^2); 
            UU_flow(indg) = 0.1;
            UU_flow_rscl  = min(UU_flow,UU_max);
            u_flow        = u_flow.*UU_flow_rscl./UU_flow;
            v_flow        = v_flow.*UU_flow_rscl./UU_flow;

            % Plot flow field magnitude with heatmap
            pcolor(-reshape(yg,Nxg,Nyg),reshape(xg,Nxg,Nyg),...
                   min(abs(reshape(UU_flow_rscl,Nxg,Nyg)),1000));

            % Plot flow field vector with quiver (reduced grid
            % optional)
            if sepgridquiv
                quiver(-ygq,xgq,-v_flow(indf2q),u_flow(indf2q),...
                    'LineWidth',1,'MaxHeadSize',0.4,'Color','w');
            else 
                quiver(-yg,xg,-v_flow,u_flow,...
                    'LineWidth',1,'MaxHeadSize',0.4,'Color','w');
            end
            shading interp; colormap parula; %caxis([0 0.5]);
            axis equal; 

            % Plot flagella shapes
            plot(-yf1,xf1,'k','LineWidth',2);    %Plot right flagellum
            plot(-yf2,xf2,'k','LineWidth',2);    %Plot left flagellum           
            
            % Plot cell body outline
            plot(-yhead,xhead,'k','LineWidth',2); hold on;              
            
            pause(eps);
            % Plot the outline of pipette+cellBody 
            [xh_plot,yh_plot] = connectAdjacentDots(xh(ih),yh(ih));
            xh_plot           = smooth(xh_plot);
            yh_plot           = smooth(yh_plot);
            plot(-yh_plot,xh_plot,'r-','LineWidth',2);           
            
            if compute_flow == 3
               plot(-ygb,xgb,'w.','Markersize',30); 
            end
            try 
                set(gca,'ylim',[ymin,ymax],'xlim',[xmin,xmax])
            catch
            end
            F(kk) = getframe(gcf);   %Write frame to movie variable
        end                                 
        if (compute_flow == 2) || (compute_flow == 3)
            M_Flow_f1b   = flagella_interaction(xf1,yf1,zf1,xgb,ygb,zgb,Nf,Nb,rf);   %Flow matrix for right flagellum
            M_Flow_f2b   = flagella_interaction(xf2,yf2,zf2,xgb,ygb,zgb,Nf,Nb,rf);   %Flow matrix for left flagellum
            UFb          = M_Flow_headb*fh(:)+M_Flow_f1b*f1(:)+M_Flow_f2b*f2(:);     %Velocity field                     [micron/s]
            Uflowb(kk,:) = UFb(1:3:end)-U(1,kk);   %x velocity     [micron/s]
            Vflowb(kk,:) = UFb(2:3:end)-U(2,kk);   %y velocity     [micron/s]
            Wflowb(kk,:) = UFb(3:3:end)-U(3,kk);   %z velocity     [micron/s]
        end
    end

    % save the velocity of flagellum over time
    flagvx(kk,1,:) = vxf1;
    flagvx(kk,2,:) = vxf2;
    flagvy(kk,1,:) = vyf1;
    flagvy(kk,2,:) = vyf2;
    flagvz(kk,1,:) = vzf1;
    flagvz(kk,2,:) = vzf2;
    clc
end


%% Log
% 2019-01-17 A copy from: Flow_Around_Chlamy_003_CalcImgSequence_CORE.m
% Code organized, commented and redundant deleted.
% same code Greta had at the end the her PhD was also taken as a reference.
% 
% User: Da

