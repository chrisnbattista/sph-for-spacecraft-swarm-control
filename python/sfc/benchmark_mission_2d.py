

N_PARTICLES = 15
LEADER = N_PARTICLES - 1


import numpy as np
import math

from multi_agent_kinetics import experiments, viz, worlds, forces
from multi_agent_kinetics import experiments, viz, worlds, forces

from uhm_hcl import state_estimator

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    initial_state=experiments.initialize_random_circle(n_particles=N_PARTICLES, radius=N_PARTICLES*1.2, min_dist=3),
    n_agents=N_PARTICLES,
    timestep=0.001,
    forces=[
        lambda x, context: forces.linear_attractor(x, 500, target=LEADER, context=context),
        lambda x, context: forces.world_pressure_force(x, h=1, pressure=0.001, context=context),
        ##lambda x: forces.world_viscosity_force(x, h=5),
        lambda x, context: forces.viscous_damping_force(x, 80, context=context),
        lambda x, context: forces.swarm_leader_force(x, np.array([200, 200]), context=context)
        ],
    context={
        'sph_active': [True] * (N_PARTICLES - 1) + [False],
        'swarm_leader': LEADER,
        'spatial_dims': 2
    }
)

agents = [state_estimator.estimator_init(np.append(world.get_state()[i, 3:5], 0)) for i in range(N_PARTICLES)]

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics")
fig2, ax2 = viz.set_up_figure(title="Benchmark Mission 1 – 2D plot")

# example data for particle id, mass, position, velocity, and time of data
sample_data = np.array([
    [0, 10, 50, 9, 1],
    [1, 10, 150, 9, 1],
    [2, 10, 300, 9, 1],
    [3, 10, 720, 9, 1],
    [4, 10, 1000, 9, 1],
])
print("Starting sim...")

for i in range(10000):

    world.advance_state()
    state = world.get_state()

    # HCL
    #neighbor_location = np.zeros((3,1))
    #std = np.zeros((3,1))

    #dead_reckoning_estimates = [state_estimator.eif_dr(agents[i], world.timestep_length, np.append(state[i,5:7], 0).T, std) for i in range(N_PARTICLES)]
    #sensor_fusion_estimates = [state_estimator.eif(agents[i], world.timestep_length, np.append(state[i, 5:7], 0).T, std, np.append(state[i:,3:5], 0).T, std, neighbor_location)]
    # loop through n^2, each neighbor of each agent (only if close enough)
    # hdj is neighbor state estimate



    if i % 50 == 0:
        viz.render_2d_orbit_state(
            state,
            fig2,
            ax2,
            agent_colors=['k']*(N_PARTICLES-1)+['b'],
            agent_sizes=[100]*N_PARTICLES
        )
        viz.render_projected_2d_orbit_state(
            state,
            fig,
            ax,
            agent_colors=['k']*(N_PARTICLES-1)+['b'],
            agent_sizes=[10]*N_PARTICLES
        )
