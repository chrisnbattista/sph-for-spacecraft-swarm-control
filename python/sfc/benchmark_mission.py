



import numpy as np

from multi_agent_kinetics import experiments, viz, worlds, forces

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    initial_state=experiments.initialize_random_circle(n_particles=5, radius=30, min_dist=7),
    n_agents=5,
    forces=[
        lambda x: forces.gravity_well(x, 200),
        lambda x: forces.world_pressure_force(x),
        lambda x: forces.world_viscosity_force(x)
        ]
)

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics")

# example data for particle id, mass, position, velocity, and time of data
sample_data = np.array([
    [0, 10, 50, 9, 1],
    [1, 10, 150, 9, 1],
    [2, 10, 300, 9, 1],
    [3, 10, 720, 9, 1],
    [4, 10, 1000, 9, 1],
])

for i in range(1000):

    world.advance_state()

    viz.render_1d_orbit_state(
        sample_data,
        fig,
        ax,
        agent_colors=['k']*4+['b'],
        agent_sizes=[100]*5
    )
