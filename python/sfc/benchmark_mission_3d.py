
import seaborn as sns
import numpy as np
import math

from multi_agent_kinetics import experiments, viz, worlds, forces, indicators, constraints

from uhm_hcl import state_estimator

print("Initiating benchmark mission...")

print("Seeding initial state...")




N_PARTICLES = 7
H = 1
if input("Enter something to use 1D benchmark >"):
    cs = [
        lambda w,s,c: constraints.linear_motion(w, s, np.array([7660, 0, 0]), context=c),
        lambda w,s,c: constraints.recenter_on_agent(w, s, LEADER, context=c),
        constraints.constrain_to_orbit
    ]
    N_PARTICLES = 5
else:
    cs = [
        ##lambda w,s,c: constraints.linear_motion(w, s, np.array([7660, 0, 0]), context=c),
        ##lambda w,s,c: constraints.recenter_on_agent(w, s, LEADER, context=c)
    ]
N_OBSTACLES = 25
LEADER = 0

world = worlds.World(
    initial_state=experiments.initialize_random_circle(
                                                        n_particles=N_PARTICLES + N_OBSTACLES,
                                                        radius=20,
                                                        min_dist=1,
                                                        random_speed=10,
                                                        spatial_dims=3),
    n_agents=N_PARTICLES + N_OBSTACLES,
    spatial_dims=3,
    n_timesteps=1000000,
    timestep=0.001,
    forces=[
        ##lambda x, context: forces.linear_attractor(x, H*50, target=LEADER, context=context),
        ##lambda x, context: forces.world_pressure_force(x, h=H, pressure=10000, context=context),
        ##lambda x: forces.world_viscosity_force(x, h=5),
        ##lambda x, context: forces.viscous_damping_force(x, H*100, context=context)
        ##lambda x, context: forces.swarm_leader_force(x, np.array([200, 0]), context=context)
        ],
    indicators=[
        indicators.total_sph_delta_v,
    ],
    constraints=cs,
    context={
        'sph_active': [True] + [True] * (N_PARTICLES - 1) + [False] * (N_OBSTACLES),
        'following_active': [True] + [True] * (N_PARTICLES - 1) + [False] * (N_OBSTACLES),
        'damping_active': [True] + [True] * (N_PARTICLES - 1) + [False] * (N_OBSTACLES),
        'swarm_leader': LEADER,
        'spatial_dims': 3
    }
)
pos = worlds.pos[world.spatial_dims]
vel = worlds.vel[world.spatial_dims]
world.control_agents = [state_estimator.Agent(world.get_state()[i, pos]) for i in range(N_PARTICLES)]

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics",
                            plot_type='2d_proj_orbit')
fig2, ax2 = viz.set_up_figure(title="Benchmark Mission 1 – 3D plot")

input("Press enter when ready >")
print("Starting sim...")

H_0 = H

for i in range(1000000):

    world.advance_state()
    state = world.get_state()

    #H = H_0 + np.sin(i/1000) / 3
    if i > 10000: H = H_0 * 2

    # HCL
    std = np.full((3,1), 1)
    #dead_reckoning_estimates = [state_estimator.eif_dr(agents[i], world.timestep_length, np.append(state[i,5:7], 0).T, std) for i in range(N_PARTICLES)]
    # for j in range(N_PARTICLES):
    #     for k in range(N_PARTICLES):
    #         if j == k: continue
    #         world.control_agents[j].eif(
    #             dt=world.timestep_length, 
    #             u=state[j, vel].T + np.random.normal(0, std, (3,1)), # velocity states
    #             sigma_u=std, 
    #             r=[state[j,pos].T - state[k,pos].T + np.random.normal(0, std, (3,1))], # difference in position states
    #             sigma_r=[std], 
    #             neighbor=state[k,pos].T + np.random.normal(0, std, (3,1)),
    #             gps=state[j,pos].T + np.random.normal(0, std, (3,1)),
    #             sigma_gps=std
    #         )
        ##world.control_agents[j].add_to_history()
                

    if i % 10 == 0:
        if i % 500 == 0: si = True
        else: si = False
        viz.render_2d_orbit_state(
            world,
            fig2,
            ax2,
            agent_colors=['g']+['k']*(N_PARTICLES-1)+['r']*(N_OBSTACLES),
            ##agent_sizes=[100]*N_PARTICLES,
            h=2*H,
            t=world.current_timestep * world.timestep_length,
            show_indicators=si,
            indicator_labels=[
                ('Total SPH force (kN)', 'time (ksec)', 'total SPH delta_v (km/ksec^2')
                ]
        )
        # viz.render_projected_2d_orbit_state(
        #     world,
        #     fig,
        #     ax,
        #     agent_colors=['g']+['k']*(N_PARTICLES-1)+['r']*(N_OBSTACLES),
        #     ##agent_sizes=[10]*N_PARTICLES,
        #     orbit_radius=6378 + 408, # radius of earth + ISS orbit altitude
        #     t=world.current_timestep * world.timestep_length
        # )
        if False:
            for a in range(len(world.control_agents)):
                # || estimated position - actual position ||2
                ##print(world.control_agents[a].history)
                error_each_step = np.linalg.norm(
                    np.stack(world.control_agents[a].history) - world.get_history()[a:world.current_timestep*world.n_agents:world.n_agents,3:5],
                    axis=1
                )
                sns.lineplot(
                    x=np.linspace(world.timestep_length, world.current_timestep*world.timestep_length, world.current_timestep),
                    y=error_each_step,
                    ax=viz.get_floating_plot(f"Agent {a} HCL Error").gca(),
                    legend=False
                )
