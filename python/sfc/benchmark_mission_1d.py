





from  multi_agent_kinetics import experiments, viz, worlds, forces

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    spatial_dims=1,
    initial_state=experiments.initialize_random_circle(n_particles=5, radius=20, min_dist=1.5, spatial_dims=1),
    n_agents=5,
    forces=[
        lambda x: forces.gravity_well(x, 200),
        lambda x: forces.world_pressure_force(x, h=1),
        ###lambda x: forces.world_viscosity_force(x, h=5),
        lambda x: forces.viscous_damping_force(x, 150)
        ]
)

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics")

print(world.get_state())

print("Starting sim...")
for i in range(1000):

    world.advance_state()

    # viz.render_state(
    #     world.get_state(),
    #     fig,
    #     ax,
    #     agent_colors=['k']*14+['b'],
    #     agent_sizes=[100]*5
    # )