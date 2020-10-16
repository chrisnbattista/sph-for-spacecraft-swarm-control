classdef sph_sim
% Class to handle vehicle control simulations using SPH.
% 
% 
% 
% The public methods include:
% 
% obj = sph_sim(param,group_conf,t0);
%           Class contructor, handles initialization.
% 
% obj = obj.sph_update_properties(param,group_conf);
%           Update the SPH parameters and/or group properties.
%
% obj = obj.sph_sim_step(rdx,lx,lR,velfunc,datapath);
%           Take a time step forward in the simulation.
% 
% time = obj.get_time()
%           Return the current time in the SPH simulation.
% 
% time = obj.get_initial_time()
%           Return the initial time for the SPH simulation.
% 
% dt = obj.get_dt();
%           Return the time step to be used for the SPH simulation.
% 
% states = obj.get_states();
%           Return a matrix containing the [x y z] positions and [u v w] of
%           all the SPH particles. Each particle is stored in one row so the
%           format is:
%                    [ x1 y1 z1 u1 v1 w1 ]
%           states = [ x2 y2 z2 u2 v2 w2 ]
%                    [        ...        ]
% 
% N = obj.get_npart();
%           Return the total number of SPH particles in the simulation.
% 
% nveh = obj.get_nveh();
%           Return the number of vehicles in the simulation.
% 
% nobs = obj.get_nobs();
%           Return the number of obstacles in the simulation.
% 
% nrd = obj.get_nrd();
%           Return the number of attractor particles in the simulation.
% 
% x = obj.get_x();
%           Return a column vector containing the x positions of all the SPH
%           particles.
%         
% y = obj.get_y();
%           Return a column vector containing the y positions of all the SPH
%           particles.
% 
% z = obj.get_z();
%           Return a column vector containing the z positions of all the SPH
%           particles.
% 
% u = obj.get_u();
%           Return a column vector containing the u-velocities (velocity in
%           the x-direction) of all SPH particles.
% 
% v = obj.get_v();
%           Return a column vector containing the v-velocities (velocity in
%           the y-direction) of all SPH particles.
% 
% w = obj.get_w();
%           Return a column vector containing the w-velocities (velocity in
%           the z-direction) of all SPH particles.
% 
% properties = obj.get_prop();
%           Returns a structure containing all the properties of each SPH
%           particle in the simulation.
% 
% param = obj.get_param();
%           Return a structure containing the parameters used in the SPH
%           simulation.
% 
% group_conf = obj.get_group_conf();
%           Return a structure containing the group configuration.
% 
% W = obj.kernel(r,h);
%           Compute the value of the smoothing kernel;
%
% dWdr = obj.kernel_grad(r,h);
%           Compute the value of the derivative of the 
%           smoothing kernel;
%

    
    properties ( Constant )
        %SPH constants
        constants = struct(...
                'particle_type',struct('veh',{1},'obs',{2},'rd',{3}), ...
                'rho0',{1}... %reference density
                );
    end
    
    properties ( Access = private )
        
        
        %loiter circle x,y,R;
        lx; lR;
        
        %time
        t; t0;
        
        %list of properties for each SPH particle
        prop;
        
        %parameters for the SPH equations
        param;
        
        %group configuration (number of vehicles, vehicle constraints, etc)
        group_conf;
        
        %particle states
        states;
        
        npart; %total number SPH particles
        nveh; %number of vehicles
        nobs; %number of vehicles
        nrd; %number of vehicles
        
    end
    
    methods ( Access = public )
    %Public methods
        
        function obj = sph_sim(param,group_conf,t0)
            % Class constructor: obj = sph_sim(param,group_conf,t0)
            %
            % Inputs:
            % param         structure containing the SPH parameters
            % group_conf    structure containing the SPH group configureation
            % t0            option, if t0 is not give it is set to 0
            %
            % Example of SPH simulation parameters:
            % param = struct(...
            %   'ndim',{2},... %dimension of the simulation (2 or 3)
            %   'gain',struct('sph',{1},'ext',{.25},'drag',{.5}),... %gain coefs
            %   'accel',struct('veh',{1},'obs',{1},'rd',{1}),... %scales accel due to vehicles/obstacles/reduced density particles
            %   'Re',{10},... %Reynolds number
            %   'dt', 0.05 ... %timestep for SPH simulation
            % );
            %             
            % Example of vehicle group configuration:
            % group_conf = struct(...
            %   'num_veh',{[9 5]},... % Array containing the number of vehicles in each group
            %   'veh_init',struct('x',{[2.5 18]},... %initial positions for the veh. groups
            %                     'y',{[7 7]},...
            %                     'z',{[0 0]},...
            %                     'u',{[0.1 0.1]},... %initial velocities for the veh. groups
            %                     'v',{[-0.1 -.1]},...
            %                     'w',{[0 0]}),...
            %   'veh_h',{[.5 1]},... % Smoothing width for each group
            %   'veh_limits',struct('vmin',{3*[1 1]},... %limits for speed and acceleration
            %                       'vmax',{5*[1 1]},...
            %                       'turning_radius',{.25*[1 1]}),...
            %   'num_obs',{2},...    % total number of obstacle particles
            %   'obs_h',{.6*[1 1]},...    % .5*size of obstacle particles
            %   'obs_init',struct('x',{[5.5 7.5]},... %positions for the obstacles
            %                     'y',{[4 6]},...
            %                     'z',{[0 0]}),...
            %   'num_rd',{0},...     % total number of reduced density particles
            %   'rd_group',{[1 2]},...% which group does each red. density particle belong to?
            %   ...                  % group number corresponds to array index for num_veh,
            %   ...                  % 0 means not active
            %   'rd_h',{10*[1 1]},...
            %   'num_loiter',{2},...     % total number of loiter circles
            %   'loiter_group',{[1 2]}...% which group does each loiter circle belong to?
            %   ...                  % group number corresponds to array index for num_veh,
            %   ...                  % 0 means not active
            % );
            %
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            
            %define parameters
            obj.param=param;
            
            %define group configuration
            obj.group_conf=group_conf;
            
            %initialize time
            if exist('t0')
                obj.t0=t0; obj.t=t0;
            else
                obj.t0=0; obj.t=0;
            end
            
            %setup SPH properties
            obj = obj.init_prop();
            obj = obj.compute_hij();
            obj = obj.kernel_type();
            
            %initialize positions/velocities
            obj = obj.init_states();
            
        end
        
        function obj = sph_update_properties(obj,param,group_conf)
        % obj = sph_update_properties(obj,param,group_conf);
        % Update or change the SPH properties to match the properties in
        % the inputs 'param' and 'group_conf'
            
            %define parameters
            obj.param=param;
            
            %define group configuration
            obj.group_conf=group_conf;
            
            %setup SPH properties
            obj = obj.init_prop();
            obj = obj.compute_hij();
            
        end
        
        function obj = sph_sim_step(obj,rdx,lx,lR,velfunc,datapath)
        % obj = sph_sim_step(obj,rdx,lx,lR,velfunc,datapath);
        % 
        % Take a single time step forward in the simulation
        % Inputs:
        % obj           an object of the sph_sim class
        % rdx = [x(:,1) y(:,1) z(:,1)]
        %               positions of the reduced density particles,
        %               optional
        % lx = [x(:,1) y(:,1) z(:,1)]
        %               positions of any loiter circles, optional
        % lR = lR(:,1)    radius of any loiter circles (optional), optional
        % velfunc       function handle to a velocity function of the form
        %                 [u v w] = velfunc(x,y,z,t,datapath), optional
        % datapath      path to data files used by velfunc, must be
        %                 included if velfunc is included
        % 
        % Omit or pass [] for any un-used inputs.
        
        
            %set reduced density particle positions
            if (exist('rdx','var') && obj.group_conf.num_rd>0)
                if size(rdx,1)~=obj.group_conf.num_rd && size(rdx,1)~=1
                    error([ ...
                         'Error in input to sph_sim_step:\n' ...
                         '%d ~= %d\n' ...
                         'Number of reduced density particle ' ...
                         'positions must equal the number of ' ...
                         'reduced density particles.' ...
                        ],size(rdx,1),obj.group_conf.num_rd)
                    
                end
                
                I1=obj.nveh+obj.group_conf.num_obs+1;
                obj.states(I1:end,1)=rdx(:,1);
                obj.states(I1:end,2)=rdx(:,2);
                if obj.param.ndim==3
                    obj.states(I1:end,3)=rdx(:,3);
                end
            end
            
            
            
            %set the loiter circle positions and radii
            if (exist('lx','var') && obj.group_conf.num_loiter>0)
                if size(lx,1)~=obj.group_conf.num_loiter
                    error(sprintf([ ...
                         'Error in input to sph_sim_step:\n' ...
                         '%d ~= %d\n' ...
                         'Number of loiter circle positions must ' ...
                         'equal the number of loiter circles.' ...
                        ],size(lx,1),obj.group_conf.num_loiter))
                end
                
                obj.lx(:,:) = lx;
                
                %set radius:
                if exist('lR','var') && ~isempty(lR)
                    obj.lR = lR;
                else
                    %radius = minimum turning radius
                    for i=1:obj.group_conf.num_loiter
                        group_num=obj.group_conf.loiter_group(i);
                        obj.lR(i) = max([ .5*obj.group_conf.veh_h(group_num)/sin(pi/max([2 obj.group_conf.num_veh(group_num)])) ...
                          obj.group_conf.veh_limits.turning_radius(group_num) ]);
                    end
                end
                
            end
            
            
            
            
            
            %assume the background velocity is changing slower than the SPH
            %timestep so just load it once
            if exist('velfunc')
                %load background velocity
                [u v w]=feval(velfunc,obj.states(:,1),obj.states(:,2),obj.states(:,3),obj.t,datapath);
            else
                if ~exist('u','var')
                    u=zeros(size(obj.states,1),1);
                    v=zeros(size(obj.states,1),1);
                    w=zeros(size(obj.states,1),1);
                end
            end
            
            
            
            
            
%             %Forward Euler time step:
%             k1 = sph_rhs(obj);
%             k1(:,1:3) = k1(:,1:3)+[u v w];
%             
%             obj.states = obj.states+obj.param.dt*k1;
%             obj.t=obj.t+obj.param.dt; %increment time
            
            
            
            
            
            
            
            
            
            
            
            %Fourth order Runge-Kutta time step:
            k1 = sph_rhs(obj);
            k1(:,1:3) = k1(:,1:3)+[u v w];
            
            tmp=obj;
            tmp.states = tmp.states+tmp.param.dt/2*k1;
            k2 = sph_rhs(tmp);
            k2(:,1:3) = k2(:,1:3)+[u v w];
            
            tmp=obj;
            tmp.states = tmp.states+tmp.param.dt/2*k2;
            k3 = sph_rhs(tmp);
            k3(:,1:3) = k3(:,1:3)+[u v w];
            
            tmp=obj;
            tmp.states = tmp.states+tmp.param.dt*k3;
            k4 = sph_rhs(tmp);
            k4(:,1:3) = k4(:,1:3)+[u v w];
            
            obj.states = obj.states+obj.param.dt/6*(k1+2*k2+2*k3+k4);
            obj.t=obj.t+obj.param.dt; %increment time
            
            
            
            
            
            
            
            
            
            %contrain the velocity:
            obj = constrain_vel(obj);
            
            
        end
        
        function time = get_time(obj)
        % time = obj.get_time();
        % Return the current time in the SPH simulation.
            time=obj.t;
        end
        
        function time = get_initial_time(obj)
        % time = obj.get__initial_time();
        % Return the initial time for the SPH simulation.
           
            time=obj.t0;
            
        end
        
        function dt = get_dt(obj)
        % dt = obj.get_dt();
        % Return the time step to be used for the SPH simulation.
            dt=obj.param.dt;
        end
        
        function states = get_states(obj)
        % states = obj.get_states();
        % Return a matrix containing the [x y z] positions and [u v w] of
        % all the SPH particles. Each particle is stored in one row so the
        % format is:
        %          [ x1 y1 z1 u1 v1 w1 ]
        % states = [ x2 y2 z2 u2 v2 w2 ]
        %          [        ...        ]
            states=obj.states;
        end
        
        function n = get_npart(obj)
        % n = obj.get_npart();
        % Return the total number of particles in the simulation.
            n=obj.npart;
        end
        
        function n = get_nveh(obj)
        % n = obj.get_nveh();
        % Return the number of vehicles in the simulation.
            n=obj.nveh;
        end
        
        function n = get_nobs(obj)
        % n = obj.get_nobs();
        % Return the number of obstacles in the simulation.
            n=obj.nveh;
        end
        
        function n = get_nrd(obj)
        % n = obj.get_nrd();
        % Return the number of reduced density (attractor) particles in the
        % simulation.
            n=obj.nrd;
        end
        
        function x = get_x(obj)
        % x = obj.get_x();
        % Return a column vector containing the x positions of all the SPH
        % particles.
            x=obj.states(:,1);
        end
        
        function y = get_y(obj)
        % y = obj.get_y();
        % Return a column vector containing the y positions of all the SPH
        % particles.
            y=obj.states(:,2);
        end
        
        function z = get_z(obj)
        % z = obj.get_z();
        % Return a column vector containing the z positions of all the SPH
        % particles.
            z=obj.states(:,3);
        end
        
        function u = get_u(obj)
        % u = obj.get_u();
        % Return a column vector containing the u-velocities (velocity in
        % the x-direction) of all SPH particles.
            u=obj.states(:,4);
        end
        
        function v = get_v(obj)
        % v = obj.get_v();
        % Return a column vector containing the v-velocities (velocity in
        % the y-direction) of all SPH particles.
            v=obj.states(:,5);
        end
        
        function w = get_w(obj)
        % w = obj.get_w();
        % Return a column vector containing the w-velocities (velocity in
        % the z-direction) of all SPH particles.
            w=obj.states(:,6);
        end
        
        function properties = get_prop(obj)
        % properties = obj.get_prop();
        % Returns a structure containing all the properties of each SPH
        % particle in the simulation. The properties include:
        % properties.vmin               minimum velocity constraint
        % properties.vmax               maximum velocity constraint
        % properties.turning_radius     turning radius constraint
        % properties.amax               maximum acceleration constraint
        % properties.h                  kernel width
        % properties.m                  mass
        % properties.mu                 viscosity
        % properties.K                  bulk modulus
        % properties.group              group number
        % properties.particle_type      particle type (veh, obs, or rd)
        % properties.hij                h_ij matrix
            properties = obj.prop;
        end
        
        function param = get_param(obj)
        % param = obj.get_param();
        % Return a structure containing the parameters used in the SPH
        % simulation. The parameters include:
        % param.ndim            dimension of the simulation (2 or 3)
        % param.gain.sph        gain coefficient for the SPH forces
        % param.gain.ext        gain coefficient for the external force
        % param.gain.drag       gain coefficient for the drag force
        % param.accel.veh       scaling constant for SPH vehicle forces
        % param.accel.obs       scaling constant for SPH obstacle forces
        % param.accel.rd    	scaling constant for SPH attractor forces
        % param.Re              Reynolds number
        % param.dt              Time step
            param=obj.param;
        end
        
        function group_conf = get_group_conf(obj)
        % group_conf = obj.get_group_conf();
        % Return a structure containing the group configuration. An example
        % is included here:
        % group_conf = struct(...
        % 'num_veh',{[4 5]},... % Array containing the number of vehicles in each group
        % 'veh_init',struct('x',{[2.5 18]},... %initial positions for the veh. groups
        %                   'y',{[7 7]},...
        %                   'z',{[0 0]},...
        %                   'u',{[0.1 0.1]},... %initial velocities for the veh. groups
        %                   'v',{[-0.1 -.1]},...
        %                   'w',{[0 0]}),...
        % 'veh_h',{[.5 1]},... % Smoothing width for each group
        % 'veh_limits',struct('vmin',{[5 0]},... %limits for speed and acceleration
        %                     'vmax',{[6 5]},...
        %                     'turning_radius',{.25*[1 1]}),...
        % 'num_obs',{2},...    % total number of obstacle particles
        % 'obs_h',{.6*[1 1]},...    % .5*size of obstacle particles
        % 'obs_init',struct('x',{[5.5 7.5]},... %positions for the obstacles
        %                   'y',{[4 6]},...
        %                   'z',{[0 0]}),...
        % 'num_rd',{0},...     % total number of reduced density particles
        % 'rd_group',{[1 2]},...% which group does each red. density particle belong to?
        % ...                  % group number corresponds to array index for num_veh,
        % ...                  % 0 means not active
        % 'rd_h',{10*[1 1]},...
        % 'num_loiter',{2},...     % total number of loiter circles
        % 'loiter_group',{[1 2]}...% which group does each loiter circle belong to?
        % ...                  % group number corresponds to array index for num_veh,
        % ...                  % 0 means not active
        % );
            group_conf=obj.group_conf;
        end
        
    end
    
    
    methods ( Access = private )
    %Private methods
        
        function obj = init_prop(obj)
            % Function to initialize the properties for each SPH particle
            % obj = obj.init_prop()
            
            N=0;
            
            %vehicles
            for i=1:length(obj.group_conf.num_veh)
                for j=1:obj.group_conf.num_veh(i)
                    N=N+1;
                    
                    %motion constraints:
                    obj.prop.vmin(N,1) = obj.group_conf.veh_limits.vmin(i);
                    obj.prop.vmax(N,1) = obj.group_conf.veh_limits.vmax(i);
                    obj.prop.turning_radius(N,1) = obj.group_conf.veh_limits.turning_radius(i);
                    obj.prop.amax(N,1) = obj.prop.vmax(N)^2/obj.prop.turning_radius(N);
                    
                    %smoothing width
                    obj.prop.h(N,1)=obj.group_conf.veh_h(i);
                    %mass
                    obj.prop.m(N,1)=obj.constants.rho0/obj.kernel(0,obj.prop.h(N),2);
                    
                    %kernel values at 0 and h:
                    KER0=obj.kernel(0,obj.prop.h(N),2);
                    KERh=obj.kernel(obj.prop.h(N),obj.prop.h(N),2);
                    
                    %kernel gradient at h
                    KERG=obj.kernel_grad(obj.prop.h(N),obj.prop.h(N),2);
                    
                    %pressure force ~ K*Fp
                    Fp=obj.constants.rho0*KERh/KER0;
                    %viscous force ~ mu*Fmu
                    Fmu=2*obj.prop.vmax(N)*KER0*KERG/( obj.constants.rho0*obj.prop.h(N)*(KER0+KERh)^2 );
                    
                    % Force coeficients found by solving:
                    %   amax = vmax^2/turning_radius = K*Fp + mu*Fmu
                    %   Re = (K*Fp)/(mu*Fmu)
                    %
                    % This enforces the desired Reynolds number at r_ij=h
                    % and limits the acceleration magnitude to be on the
                    % order of amax.
                    obj.prop.mu(N,1)=obj.prop.vmax(N)/obj.prop.turning_radius(N)/Fmu/(1+obj.param.Re);
                    obj.prop.K(N,1)=obj.param.accel.veh*obj.param.Re*obj.prop.mu(N)*Fmu/Fp;
                    
                    % group number and particle type:
                    obj.prop.group(N,1)=i;
                    obj.prop.particle_type(N,1)=obj.constants.particle_type.veh;
                end
            end
            
            obj.nveh=N;
            
            %obstacles
            for i=1:obj.group_conf.num_obs
                N=N+1;
                
                %motion constraints:
                obj.prop.vmin(N,1) = 0;
                obj.prop.vmax(N,1) = 0;
                obj.prop.turning_radius(N,1) = 0;
                obj.prop.amax(N,1) = 0;
                
                %smoothing width
                obj.prop.h(N,1)=obj.group_conf.obs_h(i);
                %mass
                obj.prop.m(N,1)=2*obj.constants.rho0/obj.kernel(0,obj.prop.h(N),2);
                
                %kernel values at 0 and h:
                KER0=obj.kernel(0,obj.prop.h(N),2);
                KERh=obj.kernel(obj.prop.h(N),obj.prop.h(N),2);
                
                %kernel gradient at h
                KERG=obj.kernel_grad(obj.prop.h(N),obj.prop.h(N),2);
                
                % Force coeficients:
                obj.prop.mu(N,1)=0;
                obj.prop.K(N,1)=obj.param.accel.obs*max(obj.prop.amax)*KER0/(obj.constants.rho0*KERh);
                
                % group number and particle type:
                obj.prop.group(N,1)=0;
                obj.prop.particle_type(N,1)=obj.constants.particle_type.obs;
                
            end
            
            obj.nobs = obj.group_conf.num_obs;
            
            %reduced density particles
            for i=1:obj.group_conf.num_rd
                N=N+1;
                
                %motion constraints:
                obj.prop.vmin(N,1) = 0;
                obj.prop.vmax(N,1) = 0;
                obj.prop.turning_radius(N,1) = 0;
                obj.prop.amax(N,1) = 0;
                
                %smoothing width
                obj.prop.h(N,1)=obj.group_conf.rd_h(i);
                %mass
                obj.prop.m(N,1)=obj.constants.rho0/obj.kernel(0,obj.prop.h(N),1)*1e-8;
                
                % Force coeficients:
                obj.prop.mu(N,1)=0; %no viscosity for attractors
                I=find(obj.prop.group(1:obj.nveh)==obj.group_conf.rd_group(i));
                obj.prop.K(N,1)=-obj.param.accel.rd*max(obj.prop.amax(I)) ...
                        *obj.kernel(0,obj.prop.h(N),1)/obj.kernel_grad(obj.prop.h(N),obj.prop.h(N),1);
                
                % group number and particle type:
                obj.prop.group(N,1)=obj.group_conf.rd_group(i);
                obj.prop.particle_type(N,1)=obj.constants.particle_type.rd;
            end
            
            obj.nrd=obj.group_conf.num_rd;
            
            obj.npart=N; %total number of SPH particles
            
        end
        
        function obj = compute_hij(obj)
            
            hi = obj.prop.h(1:obj.nveh)*ones(1,obj.nveh);
            hj = hi';
            
            %vehicles: hij=max(hi,hj)
            obj.prop.hij = max(hi,hj);
            
            %obstacles: hij=h_obs
            I1=obj.nveh+1;
            I2=obj.nveh+obj.group_conf.num_obs;
            obj.prop.hij = [obj.prop.hij; ...
                obj.prop.h(I1:I2)*ones(1,size(obj.prop.hij,2)) ];
            obj.prop.hij = [obj.prop.hij ...
                ones(size(obj.prop.hij,1),1) ...
                * obj.prop.h(I1:I2)' ];
            
            %reduced density particles: hij=h_rd
            I1=obj.nveh+obj.group_conf.num_obs+1;
            I2=obj.nveh+obj.group_conf.num_obs+obj.group_conf.num_rd;
            obj.prop.hij = [obj.prop.hij; ...
                obj.prop.h(I1:I2)*ones(1,size(obj.prop.hij,2)) ];
            obj.prop.hij = [obj.prop.hij ...
                ones(size(obj.prop.hij,1),1) ...
                * obj.prop.h(I1:I2)' ];
            
        end
        
        function obj = kernel_type(obj)
        % create a matrix kernel_type that tells which kernel to use, 1 is for
        % vehicle-reduced density particle interactions, 2 is for all
        % others
        
            N=length(obj.prop.m);
            
            ki=obj.prop.particle_type*ones(1,N);
            kj=ki';
            
            obj.prop.kernel_type=2*ones(N);
            
            I=find(ki==obj.constants.particle_type.veh*ones(N) & kj==obj.constants.particle_type.rd*ones(N));
            obj.prop.kernel_type(I)=1;
            I=find(kj==obj.constants.particle_type.veh*ones(N) & ki==obj.constants.particle_type.rd*ones(N));
            obj.prop.kernel_type(I)=1;
            
        end
        
        function obj = init_states(obj)
        % set the initial SPH states (positions and velocities) for all
        % particles
            
            if obj.param.ndim==2
                %2d initialization
                obj = obj.init2d();
            else
                %3d initialization
                obj = obj.init3d();
            end
            
            %obstacles:
            for i=1:obj.group_conf.num_obs
                obj.states(end+1,:) = [ ...
                    obj.group_conf.obs_init.x(i) ...
                    obj.group_conf.obs_init.y(i) ...
                    obj.group_conf.obs_init.z(i) ...
                    0 0 0 ...
                                      ];
            end
            
            %reduced density particles:
            obj.states=[obj.states; ...
                zeros(obj.group_conf.num_rd,6); ...
                ];
            
        end
        
        function obj = init2d(obj)
        % 2D initialization, use a 2D hexagonal (close packed) lattice
            
            %basis vectors for a hexagonal lattice in 2D
            v1 = [1 0 0];
            v2 = [cos(pi/3) sin(pi/3) 0];

            N=90;
            %row 1 in x-dir
            r = [ 0:N-1; zeros(2,N)];
            n = size(r,2);
            for i=1:N-1
                if mod(i,2)==0
                    r = [r(1,:) r(1,end-n+1:end)+v2(1);
                         r(2,:) r(2,end-n+1:end)+v2(2);
                         r(3,:) r(3,end-n+1:end)+v2(3)];
                else
                    r = [r(1,:) r(1,end-n+1:end)+v2(1)-v1(1);
                         r(2,:) r(2,end-n+1:end)+v2(2)-v1(2);
                         r(3,:) r(3,end-n+1:end)+v2(3)-v1(3)];
                end
            end
            
            % randomize slightly to avoid singularities
            r(1:2,:) = r(1:2,:)+(rand(size(r(1:2,:)))-.5)*1e-8;
            
            %shift to origin
            ave = sum(r,2)/size(r,2);
            r(1,:) = r(1,:)-ave(1);
            r(2,:) = r(2,:)-ave(2);
            r(3,:) = 0;

            %sort r by distance from the origin
            d = sqrt(sum(r.^2,1));
            [Y I] = sort(d);
            r = r(:,I);

            %shift to origin
            ave = r(:,1);
            r(1,:) = r(1,:)-ave(1);
            r(2,:) = r(2,:)-ave(2);
            r(3,:) = 0;

            %sort r by distance from the origin
            d = sqrt(sum(r.^2,1));
            [Y I] = sort(d);
            r = r(:,I);

            %shift to origin
            r(1,:) = r(1,:)-r(1,1);
            r(2,:) = r(2,:)-r(1,2);
            r(3,:) = 0;


            r=r';
            obj.states=[];
            
            %set the initial positions:
            for i=1:length(obj.group_conf.veh_init.x)
                
                if length(obj.group_conf.veh_init.x)<length(obj.group_conf.num_veh)
                    n = sum(obj.group_conf.num_veh);
                    one_loop=true;
                else
                    n = obj.group_conf.num_veh(i);
                    one_loop=false;
                end

                xmid = obj.group_conf.veh_init.x(i);
                ymid = obj.group_conf.veh_init.y(i);
                zmid = 0;

                vx = obj.group_conf.veh_init.u(i);
                vy = obj.group_conf.veh_init.v(i);
                vz = 0;

                rtmp(:,1:3)=r(:,1:3)*2*obj.group_conf.veh_h(i);

                rtmp(:,4)=vx;
                rtmp(:,5)=vy;
                rtmp(:,6)=0;

                rn = rtmp(1:n,:);

                xave=sum(rn(:,1))/n;
                yave=sum(rn(:,2))/n;
                zave=0;

                rn(:,1)=rn(:,1)-xave+xmid;
                rn(:,2)=rn(:,2)-yave+ymid;
                rn(:,3)=0;

                obj.states = [obj.states; rn];
                
                if one_loop
                    break
                end
                
            end
            
            
        end
        
        function obj = init3d(obj)
        % 3D initialization, use a 3D hexagonal (close packed) lattice
            
            %basis vectors for a hexagonal latice in 3D
            v1 = [1 0 0];
            v2 = [cos(pi/3) sin(pi/3) 0];
            v3 = [cos(pi/3) sin(pi/3)/3 sqrt(1-cos(pi/3)^2-sin(pi/3)^2/9)];

            N=20;
            %row 1 in x-dir
            r = [ 0:N-1; zeros(2,N)];
            n = size(r,2);
            for i=1:N-1
                if mod(i,2)==0
                    r = [r(1,:) r(1,end-n+1:end)+v2(1);
                         r(2,:) r(2,end-n+1:end)+v2(2);
                         r(3,:) r(3,end-n+1:end)+v2(3)];
                else
                    r = [r(1,:) r(1,end-n+1:end)+v2(1)-v1(1);
                         r(2,:) r(2,end-n+1:end)+v2(2)-v1(2);
                         r(3,:) r(3,end-n+1:end)+v2(3)-v1(3)];
                end
            end

            n=size(r,2);
            for i=1:N-1
                if mod(i,3)==1
                    r = [r(1,:) r(1,end-n+1:end)+v3(1);
                         r(2,:) r(2,end-n+1:end)+v3(2);
                         r(3,:) r(3,end-n+1:end)+v3(3)];
                elseif mod(i,3)==2
                    r = [r(1,:) r(1,end-n+1:end)+v3(1)-v1(1);
                         r(2,:) r(2,end-n+1:end)+v3(2)-v1(2);
                         r(3,:) r(3,end-n+1:end)+v3(3)-v1(3)];
                else
                    r = [r(1,:) r(1,end-n+1:end)+v3(1)-v2(1);
                         r(2,:) r(2,end-n+1:end)+v3(2)-v2(2);
                         r(3,:) r(3,end-n+1:end)+v3(3)-v2(3)];
                end
            end
            
            % randomize slightly to avoid singularities
            r = r+(rand(size(r))-.5)*1e-8;

            %shift to origin
            ave = sum(r,2)/size(r,2);
            r(1,:) = r(1,:)-ave(1);
            r(2,:) = r(2,:)-ave(2);
            r(3,:) = r(3,:)-ave(3);

            %sort r by distance from the origin
            d = sqrt(sum(r.^2,1));
            [Y I] = sort(d);
            r = r(:,I);

            %shift to origin
            ave = r(:,1);
            r(1,:) = r(1,:)-ave(1);
            r(2,:) = r(2,:)-ave(2);
            r(3,:) = r(3,:)-ave(3);

            %sort r by distance from the origin
            d = sqrt(sum(r.^2,1));
            [Y I] = sort(d);
            r = r(:,I);

            %shift to origin
            r(1,:) = r(1,:)-r(1,1);
            r(2,:) = r(2,:)-r(1,2);
            r(3,:) = r(3,:)-r(1,3);


            r=r';
            obj.states=[];
            
            %set the initial positions:
            for i=1:length(obj.group_conf.veh_init.x)
                
                if length(obj.group_conf.veh_init.x)<length(obj.group_conf.num_veh)
                    n = sum(obj.group_conf.num_veh);
                    one_loop=true;
                else
                    n = obj.group_conf.num_veh(i);
                    one_loop=false;
                end

                xmid = obj.group_conf.veh_init.x(i);
                ymid = obj.group_conf.veh_init.y(i);
                zmid = obj.group_conf.veh_init.z(i);

                vx = obj.group_conf.veh_init.u(i);
                vy = obj.group_conf.veh_init.v(i);
                vz = obj.group_conf.veh_init.w(i);

                rtmp(:,1:3)=r(:,1:3)*2*obj.group_conf.veh_h(i);

                rtmp(:,4)=vx;
                rtmp(:,5)=vy;
                rtmp(:,6)=vy;

                rn = rtmp(1:n,:);

                xave=sum(rn(:,1))/n;
                yave=sum(rn(:,2))/n;
                zave=sum(rn(:,3))/n;

                rn(:,1)=rn(:,1)-xave+xmid;
                rn(:,2)=rn(:,2)-yave+ymid;
                rn(:,3)=rn(:,3)-zave+zmid;

                obj.states = [obj.states; rn];
                
                if one_loop
                    break
                end
                
            end
            
        end
        
        function rhs = sph_rhs(obj)
            % Return the right hand side of the SPH momentum equation

            %compute the interparticle distance and vectors
            [dij rij unit_ij] = obj.sph_compute_dij();

            %compute the mask function Mask(i,j)=1 if particles interact
            [Mask MaskI] = obj.sph_compute_mask(dij);

            %compute density
            rho = obj.sph_compute_density(dij,Mask,MaskI);

            %remove diagonal elements from MaskI
            tmp = Mask-sparse(eye(obj.npart));
            MaskI = find(tmp==1);

            %compute gradW
            gradW = sparse([],[],[],obj.npart,obj.npart,length(MaskI));
            gradW(MaskI) = obj.kernel_grad(dij(MaskI),obj.prop.hij(MaskI),obj.prop.kernel_type(MaskI));

            %compute pressure
            P = obj.sph_compute_pressure(rho);
            P_term = (P./rho.^2)*obj.prop.m'+ ones(obj.npart,1)*(P.*obj.prop.m./rho.^2)';
            P_term = P_term.*gradW; %magnitude of the pressure force

            %viscosity
            Pi_term = obj.sph_compute_pi(rho,dij,rij,unit_ij,gradW,Mask,MaskI);

            DvDt = -cat(2,squeeze(sum(P_term.*unit_ij(:,:,1),2)),squeeze(sum(P_term.*unit_ij(:,:,2),2)),squeeze(sum(P_term.*unit_ij(:,:,3),2)));
            DvDt = DvDt+(1/obj.param.Re)*Pi_term;


            %external forcing:
            [Fx Fy Fz] = obj.external_force();
            
            %sum of forces
            DvDt =   obj.param.gain.sph*squeeze(DvDt) ...
                   + obj.param.gain.ext*max(obj.prop.amax(1:obj.nveh))*[Fx Fy Fz] ...
                   - obj.param.gain.drag.*obj.states(:,4:6);


            rhs = obj.sph_compute_rates(DvDt);



            if obj.param.ndim==2
                rhs(:,3)=0;
                rhs(:,6)=0;
            end

        end
        
        function [dij rij unit_ij]=sph_compute_dij(obj)
            %function [dij rij unit_ij]=SPH_compute_dij(states,group_conf)
            %compute the distance, vector and unit vector between particles i and j

            %create distance matrix for dij(i,j) = distance between particles i and j
            dx = obj.states(:,1)*ones(1,obj.npart);
            dx = dx-dx';
            dy = obj.states(:,2)*ones(1,obj.npart);
            dy = dy-dy';
            dz = obj.states(:,3)*ones(1,obj.npart);
            dz = dz-dz';

            dij = sqrt(dx.^2+dy.^2+dz.^2);

            rij(:,:,1)=dx;
            rij(:,:,2)=dy;
            rij(:,:,3)=dz;

            unit_ij = cat(3,rij(:,:,1)./dij,rij(:,:,2)./dij,rij(:,:,3)./dij);
            unit_ij(isnan(unit_ij))=0;

        end
        
        function [M I] = sph_compute_mask(obj,dij)
            % Compute the masking function and the indicies of M that are non-zero
            % 
            %          { 0 if dij>2*hij
            % M(i,j) = { 0 if particle j is not a vehicle and group(i)~=group(j)
            %          { 0 if particle i is not a vehicle (except M(i,i)=1)
            %          { 1 else
            
            %Kernel is nonzero (i.e. dij<2*hij):
            M = sparse(dij(1:obj.nveh,:)<2*obj.prop.hij(1:obj.nveh,:));
            
            %obstacle or reduced density particle
            M(obj.nveh+1:obj.npart,:)=0; % M(i,:)=0
            n=obj.nobs+obj.nrd;
            M(obj.nveh+1:end,obj.nveh+1:end)=spdiags(ones(n,1),0,n,n); %M(i,i)=1
            
            %reduced density particles:
            for i=1:obj.nrd
                I1=obj.nveh+obj.nobs+i;
                I2=find(obj.prop.group~=obj.prop.group(I1)); %group(i)~=group(j)
                M(I2,I1)=0; 
            end
            
            I=find(M~=0); %indices of nonzeros

        end
        
        function rho = sph_compute_density(obj,dij,Mask,MaskI)

            %reshape mass vector into matrix
            mj=Mask.*(ones(obj.npart,1)*obj.prop.m');

            K=sparse([],[],[],obj.npart,obj.npart,length(MaskI));
            K(MaskI) = obj.kernel(dij(MaskI),obj.prop.hij(MaskI),obj.prop.kernel_type(MaskI));

            rho=full(sum( mj.*K ,2));
            
            %reduced density particles have fixed density that does not
            %consider the proximity of other particles
            I = obj.nveh+obj.nobs+1:obj.npart;
            rho(I) = obj.prop.m(I).*obj.kernel(0,obj.prop.h(I),2);

        end
        
        function P = sph_compute_pressure(obj,rho)
            % Equation of state to compute the pressure

            P = obj.prop.K.*rho.*( rho/obj.constants.rho0 - 1 );

        end
        
        function Pi = sph_compute_pi(obj,rho,dij,rij,unit_ij,gradW,Mask,MaskI)
            % compute the viscous forces

            tmp = ((1./rho)*(2*obj.prop.m./rho)').*gradW;
            tmp(MaskI) = tmp(MaskI)./dij(MaskI);

            vji = cat(3,ones(obj.npart,1)*obj.states(:,4)',ones(obj.npart,1)*obj.states(:,5)',ones(obj.npart,1)*obj.states(:,6)')-cat(3,obj.states(:,4)*ones(1,obj.npart),obj.states(:,5)*ones(1,obj.npart),obj.states(:,6)*ones(1,obj.npart));

            %no viscosity for reduced density particles or obstacles
            vji(:,obj.nveh+1:end,:)=0;
            
            Pi = -cat(2,squeeze(sum(tmp.*vji(:,:,1),2)),squeeze(sum(tmp.*vji(:,:,2),2)),squeeze(sum(tmp.*vji(:,:,3),2)));

        end
        
        function [Fx Fy Fz] = external_force(obj)
        % [Fx Fy] = external_force(obj)
        % Compute the external force on vehicles to drive them toward a loiter
        % circle.
            
            Fx=zeros(size(obj.states,1),1);
            Fy=Fx; Fz=Fx;
            for i=1:obj.group_conf.num_loiter
                group_num=obj.group_conf.loiter_group(i);
                II=find(obj.prop.group==group_num);
                
                
                if obj.lR(i)>0 %loiter circle:
                    
                    %width of the "flat spot" in the potential field, controls the width of the
                    %loiter circle track
                    width=obj.lR(i)/4;

                    %shift the center of the loiter circle
                    x=obj.states(II,1)-obj.lx(i,1);
                    y=obj.states(II,2)-obj.lx(i,2);

                    %attraction component
                    d = sqrt(x.^2+y.^2);
                    d = (d-obj.lR(i))/width;
                    mag = -(tanh(d)+d./cosh(d).^2);

                    rr=sqrt(x.^2+y.^2);
                    F1x = mag.*x./rr;
                    F1y = mag.*y./rr;


                    %circulation component
                    theta = atan2(y,x);
                    F2x = -( exp(2)*(rr/obj.lR(i)).^2.*exp(-2*rr/obj.lR(i)) ).*sin(theta);
                    F2y = ( exp(2)*(rr/obj.lR(i)).^2.*exp(-2*rr/obj.lR(i)) ).*cos(theta);


                    %total force
                    w=1;
                    Fx(II) = w*F1x+(2-w)*F2x;
                    Fy(II) = w*F1y+(2-w)*F2y;
                    
                else %simple attractor (no circulation force)
                    
                    width=sqrt( length(II)*mean(obj.prop.h(II))^2 )/2;%shift the center of the loiter circle
                    x=obj.states(II,1)-obj.lx(i,1);
                    y=obj.states(II,2)-obj.lx(i,2);

                    %attraction component
                    d = sqrt(x.^2+y.^2);
                    d = (d)/width;
                    mag = -(tanh(d)+d./cosh(d).^2);

                    rr=sqrt(x.^2+y.^2);
                    Fx(II) = mag.*x./rr;
                    Fy(II) = mag.*y./rr;
                    
                end
                
            end
            
            Fx(isnan(Fx))=0;
            Fy(isnan(Fy))=0;
            
        end
        
        function rates = sph_compute_rates(obj,DvDt)
        % compute the rate of change of SPH.states, i.e. the velocity and
        % accelerations, while applying vehicle constraints

            %break acceleration into 2 components, normal and tangential:

            v = obj.states(:,4:6);%[states(4:6:end) states(5:6:end) states(6:6:end)];
            vmag = sqrt(sum(v.^2,2));
            vhat = v./(vmag*ones(1,3)); %unit vector in the v direction
            I=find(vmag==0);
            vhat(I,1)=1;
            vhat(I,2)=0;
            vhat(I,3)=0;

            
            %acceleration in the normal and tangent direction
            a_tan = ( sum(DvDt.*vhat,2)*ones(1,3) ).*vhat;
            a_norm = DvDt-a_tan;
            a_tan_mag = sqrt(sum(a_tan.^2,2));
            
            
            %limit acceleration:
            I=find(a_tan_mag>obj.prop.amax);
            if ~isempty(I)
                a_tan(I,:) = a_tan(I,:)./( a_tan_mag(I)*ones(1,3) ).*( obj.prop.amax(I)*ones(1,3) );
            end
            
            
            %limit speed
            I=find(vmag>obj.prop.vmax);
            if ~isempty(I)
                a_tan(I,:) = -obj.prop.amax(I)*ones(1,3).*vhat(I,:);
            end
            I=find(vmag<obj.prop.vmin);
            if ~isempty(I)
                a_tan(I,:) = obj.prop.amax(I)*ones(1,3).*vhat(I,:);
            end

            
            %limit turning radius
            a_norm_mag = sqrt(sum(a_norm.^2,2));
            I=find(a_norm_mag>vmag.^2./obj.prop.turning_radius);
            if ~isempty(I)
                a_norm(I,:) = a_norm(I,:)./(a_norm_mag(I)*ones(1,3)).*(vmag(I).^2./obj.prop.turning_radius(I)*ones(1,3));
            end

            rates = [obj.states(:,4:6) ( a_tan(:,1:3)+a_norm(:,1:3) )];

        end
        
        function obj = constrain_vel(obj)
        %apply velocity constraints
            
            %max vel constraint:
            V = sqrt(sum(obj.states(:,4:6).^2,2));
            I = find(V>obj.prop.vmax);
            if ~isempty(I)
                obj.states(I,4) = obj.states(I,4)./V(I).*obj.prop.vmax(I);
                obj.states(I,5) = obj.states(I,5)./V(I).*obj.prop.vmax(I);
                obj.states(I,6) = obj.states(I,6)./V(I).*obj.prop.vmax(I);
            end
            
            %min vel constraint:
            V = sqrt(sum(obj.states(:,4:6).^2,2));
            I = find(V<obj.prop.vmin);
            if ~isempty(I)
                obj.states(I,4) = obj.states(I,4)./V(I).*obj.prop.vmin(I);
                obj.states(I,5) = obj.states(I,5)./V(I).*obj.prop.vmin(I);
                obj.states(I,6) = obj.states(I,6)./V(I).*obj.prop.vmin(I);
            end
            
        end
        
    end
    
    methods ( Static )
        
        function W=kernel(r,h,type)
        % W = sph_sim.kernel(r,h,type);
        % Evaluate the smoothing kernel function for the SPH equations. 
            
            s=r./h;
            
            W=zeros(size(s));
            
            % Cubic spline kernel, used for vehicle-reduced density (type
            % 1) particle interactions.
            W = W + ( type==1 ).*( ( 1 - 3/2*s.^2 + 3/4*s.^3 ).*( s<1 ) + ( 1/4*(2-s).^3 ).*( (s >= 1).*(s <= 2) ) )./(pi*h.^3);
            
            % Quadratic kernel, used for all other (type 2) interactions
            W = W + ( type==2 ).*15./(16*pi*h.^3).*(s.^2/4-s+1).*(s<2);
            
        end
        
        function dWdr=kernel_grad(r,h,type)
        % dWdr = sph_sim.kernel_grad(r,h,type);
        % Evaluate the derivative of the smoothing kernel function for the
        % SPH equations. Note that this returns a scalar value which is the
        % magnitude of the gradient of W, the direction (if needed) must be
        % computed separately.
            
            s=r./h;
            
            dWdr=zeros(size(s));
            
            % Cubic spline kernel, used for vehicle-reduced density (type
            % 1) particle interactions.
            dWdr = dWdr + ( type==1 ).*( ( -3*s + 9/4*s.^2 ).*( s<1 ) + ( -3/4*(2-s).^2 ).*( (s >= 1).*(s <= 2) ) )./(pi*h.^4);
            
            % Quadratic kernel, used for all other (type 2) interactions
            dWdr = dWdr + ( type==2 ).*15./(16*pi*h.^4).*(s/2-1).*(s<2);
            
        end
        
    end
    
end