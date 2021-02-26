

N_PARTICLES = 15
LEADER = N_PARTICLES - 1
H = 1


import numpy as np
import math

from multi_agent_kinetics import experiments, viz, worlds, forces, indicators, constraints

from uhm_hcl import state_estimator

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    initial_state=experiments.initialize_random_circle(n_particles=N_PARTICLES, radius=N_PARTICLES*1.2, min_dist=2),
    n_agents=N_PARTICLES,
    n_timesteps=1000000,
    timestep=0.001,
    forces=[
        lambda x, context: forces.linear_attractor(x, 500, target=LEADER, context=context),
        lambda x, context: forces.world_pressure_force(x, h=H, pressure=0.001, context=context),
        ##lambda x: forces.world_viscosity_force(x, h=5),
        lambda x, context: forces.viscous_damping_force(x, 100, context=context)
        ##lambda x, context: forces.swarm_leader_force(x, np.array([200, 0]), context=context)
        ],
    indicators=[
        indicators.total_sph_force
    ],
    constraints=[
        lambda w,s: constraints.linear_motion(w, s, np.array([800, 0])),
        lambda w,s: constraints.recenter_on_agent(w, s, LEADER)
    ],
    context={
        'sph_active': [True] * (N_PARTICLES - 1) + [False],
        'swarm_leader': LEADER,
        'spatial_dims': 2
    }
)

agents = [state_estimator.estimator_init(np.append(world.get_state()[i, 3:5], 0)) for i in range(N_PARTICLES)]

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics",
                            plot_type='2d_proj_orbit')
fig2, ax2 = viz.set_up_figure(title="Benchmark Mission 1 – 2D plot")

print("Starting sim...")

for i in range(10000):

    world.advance_state()
    state = world.get_state()

    # HCL
    neighbor_location = np.zeros((3,1))
    #std = np.zeros((3,1))

    #dead_reckoning_estimates = [state_estimator.eif_dr(agents[i], world.timestep_length, np.append(state[i,5:7], 0).T, std) for i in range(N_PARTICLES)]
    #sensor_fusion_estimates = [state_estimator.eif(agents[i], world.timestep_length, np.append(state[i, 5:7], 0).T, std, np.append(state[i:,3:5], 0).T, std, neighbor_location)]
    # loop through n^2, each neighbor of each agent (only if close enough)
    # hdj is neighbor state estimate



    if i % 10 == 0:
        if i % 1000 == 0: si = True
        else: si = False
        viz.render_2d_orbit_state(
            world,
            fig2,
            ax2,
            agent_colors=['k']*(N_PARTICLES-1)+['b'],
            agent_sizes=[100]*N_PARTICLES,
            h=2*H,
            t=world.current_timestep * world.timestep_length,
            show_indicators=si,
            indicator_labels=['Total SPH force (kN)']
        )
        viz.render_projected_2d_orbit_state(
            world,
            fig,
            ax,
            agent_colors=['k']*(N_PARTICLES-1)+['b'],
            agent_sizes=[10]*N_PARTICLES,
            orbit_radius=408,
            t=world.current_timestep * world.timestep_length
        )
