function plot_veh(h,SPH,x,y,t,lx, obx)
% function plot_veh(h)
%
% Plot the vehicle and attractor locations
% Inputs:
% h         figure number to plot into, use 99 to plot on the concentration
%               plot
% SPH       object of class sph_sim
% x,y       the paths of the vehicles
% t         vector of times corresponding to the points in x and y
% lx        loiter circle location [x(:,1) y(:,1)]
% obx       obstacle locations [x y]

%what time range should be used for the track?
t_range=3;


%plot
figure(h)
clf
hold on

% %buildings
% fill([5 6 6 5],[3.5 3.5 4.5 4.5],'k')
% fill(2+[5 6 6 5],2+[3.5 3.5 4.5 4.5],'k')


%reduced density particles:
plot(x(end-SPH.get_nrd+1:end),y(end-SPH.get_nrd+1:end),'gx','linewidth',2)

%loiter circles:
if exist('lx') & SPH.get_group_conf.num_loiter>0
    plot(lx(:,1),lx(:,2),'ko','linewidth',1,'markersize',6)
end

%obstacles:
if exist('obx') & SPH.get_group_conf.num_obs>0
    plot(obx(:,1),obx(:,2),'k*','linewidth',1,'markersize',15)
end

% plot(x',y','k')
%vehicle tracks:
I=find(t>=SPH.get_time()-t_range);
for ii=1:sum(SPH.get_nveh)
    plot(x(ii,I),y(ii,I),'k-','linewidth',1)
end

%vehicles:
c=lines(length(SPH.get_group_conf.num_veh));
prop=SPH.get_prop();
for i=1:length(SPH.get_group_conf.num_veh)
    I=find(SPH.get_prop.group(1:SPH.get_nveh)==i);
    plot(x(I,end),y(I,end),'o','linewidth',2,'color',c(i,:))
end

axis equal
axis([0 30 -7 13])
title(sprintf('Smoothed Particle Hydrodynamics for Agent Control\nTime = %1.1f',SPH.get_time()),'fontsize',14,'fontname','times')
set(gca,'fontname','times')

drawnow

end

