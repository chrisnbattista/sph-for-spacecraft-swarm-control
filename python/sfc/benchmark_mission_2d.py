



import numpy as np

from multi_agent_kinetics import experiments, viz, worlds, forces
from  multi_agent_kinetics import experiments, viz, worlds, forces

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    initial_state=experiments.initialize_random_circle(n_particles=5, radius=20, min_dist=3),
    n_agents=5,
    forces=[
        lambda x: forces.gravity_well(x, 500),
        lambda x: forces.world_pressure_force(x, h=1),
        ##lambda x: forces.world_viscosity_force(x, h=5),
        lambda x: forces.viscous_damping_force(x, 20)
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
print("Starting sim...")

for i in range(1000):

    world.advance_state()

    viz.render_1d_orbit_state(
        sample_data,
    viz.render_2d_orbit_state(
        world.get_state(),
        fig,
        ax,
        agent_colors=['k']*4+['b'],
        agent_sizes=[100]*5
    )
