% Main SPH code:

clear; clc




%%%%%%%%%%%%%
% SPH setup %
%%%%%%%%%%%%%


%SPH simulation parameters
param = struct(...
'ndim',{2},... %dimension of the simulation (2 or 3)
'gain',struct('sph',{1},'ext',{.25},'drag',{10}),... %gain coefs
'accel',struct('veh',{1},'obs',{1},'rd',{.25}),... %scales accel due to vehicles/obstacles/reduced density particles
'Re',{10},... %Reynolds number
'dt', 0.01 ... %timestep for SPH simulation
);


%The groups of vehicle, obstacles, and reduced density particles
%(dimensional parameters)
group_conf = struct(...
'num_veh',{[4 5]},... % Array containing the number of vehicles in each group
'veh_init',struct('x',{[2.5 18]},... %initial positions for the veh. groups
                  'y',{[7 7]},...
                  'z',{[0 0]},...
                  'u',{[0.1 0.1]},... %initial velocities for the veh. groups
                  'v',{[-0.1 -.1]},...
                  'w',{[0 0]}),...
'veh_h',{[1 1]},... % Smoothing width for each group
'veh_limits',struct('vmin',{[3 0]},... %limits for speed and acceleration
                    'vmax',{[6 5]},...
                    'turning_radius',{.25*[1 1]}),...
'num_obs',{0},...    % total number of obstacle particles
'obs_h',{.6*[1 1]},...    % .5*size of obstacle particles
'obs_init',struct('x',{[5.5 7.5]},... %positions for the obstacles
                  'y',{[4 6]},...
                  'z',{[0 0]}),...
'num_rd',{0},...     % total number of reduced density particles
'rd_group',{[1 2]},...% which group does each red. density particle belong to?
...                  % group number corresponds to array index for num_veh,
...                  % 0 means not active
'rd_h',{10*[1 1]},...
'num_loiter',{0},...     % total number of loiter circles
'loiter_group',{[1 2]}...% which group does each loiter circle belong to?
...                  % group number corresponds to array index for num_veh,
...                  % 0 means not active
);

%initialize the SPH simulation
SPH = sph_sim(param,group_conf);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% i=0;
% plot_veh(1,SPH,SPH.states(:,1),SPH.states(:,2),0,lx,ly)
% set(gca,'position', [0.1300    0.1500    0.7750    0.8150])
% export_fig(sprintf('images/img%04d.jpg',i+1),'-q100','-transparent','-r90','-nocrop')

x=SPH.get_x(); y=SPH.get_y();
u=SPH.get_u(); v=SPH.get_v();
trackt=SPH.get_initial_time(); %used for plotting the vehicle paths
plotdt=0.1;plott=SPH.get_time();
t0=SPH.get_time();tf=100;
for t=t0:SPH.get_dt():tf-SPH.get_dt()
    
    %loiter circle locations:
    lx = [12 3; ... %first loiter circle position [x y]
          25 5]; %second loiter circle position [x y]
      
    %loiter circle radii
    if group_conf.num_loiter>0
        if SPH.get_time()<15
            lR=[9 2]; %loiter circle radii
        else
            %change the loiter circle radii at each time step
            lR=[2.5+cos((SPH.get_time()-40)/2) 3-cos(SPH.get_time()-40)];

            %update the SPH properties
            group_conf.veh_h=2*lR.*sin(pi./group_conf.num_veh);
            SPH=SPH.sph_update_properties(param,group_conf);
        end
    else
        lR=[];
    end
    
    %reduced density targets for the vehicles:
    rdx = [12 3; ... %first rd position [x y]
          25 5]; %second rd position [x y]
    
    %take an SPH timestep
    SPH=SPH.sph_sim_step(rdx,lx,lR);
    
    %keep track of vehicle paths (or at least the last 100 points)
    x=[x SPH.get_x()];
    y=[y SPH.get_y()];
    u=[u SPH.get_u()];
    v=[v SPH.get_v()];
    trackt=[trackt SPH.get_time()];
    
    if any(isnan(x(:)))>0
        error('Something went wrong, NaN detected ins x-positions.');
    end
    
    %plot
    if SPH.get_time()>=plott-SPH.get_dt()/10
        plot_veh(1,SPH,x,y,trackt,lx)
        plott=plott+plotdt;
        
%         %compute/plot the minimum distance between vehicles in each group:
%         I1=1;
%         states=SPH.get_states();
%         for i=1:length(SPH.get_group_conf.num_veh)
%             n=SPH.get_group_conf.num_veh(i);
%             I2=I1+n-1;
%             
%             dx = states(I1:I2,1)*ones(1,n);
%             dx=dx-dx';
%             dy = states(I1:I2,2)*ones(1,n);
%             dy=dy-dy';
%             dz = states(I1:I2,3)*ones(1,n);
%             dz=dz-dz';
%             dij=sqrt(dx.^2+dy.^2+dz.^2)+1e8*eye(n);
%             
%             d(i)=min(min(dij));
%             
%             I1=I2+1;
%         end
%         figure(2)
%         hold on
%         plot(SPH.get_time(),d,'.')
%         title('Minimum intervehicle spacing')
        
    end
    
% i=i+1;
% if mod(i,4)==0
% set(gca,'position', [0.1300    0.1500    0.7750    0.8150])
% export_fig(sprintf('images/img%04d.jpg',i/4+1),'-q100','-transparent','-r90','-nocrop')
% end
    
end

