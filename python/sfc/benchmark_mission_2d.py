from  multi_agent_kinetics import experiments, viz, worlds, forces

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    initial_state=experiments.initialize_random_circle(n_particles=5, radius=15, min_dist=3),
    n_agents=5,
    timestep=0.001,
    forces=[
        lambda x, context: forces.gravity_well(x, 500, context=context),
        lambda x, context: forces.world_pressure_force(x, h=1, pressure=0.00001, context=context),
        ##lambda x: forces.world_viscosity_force(x, h=5),
        lambda x, context: forces.viscous_damping_force(x, 80, context=context)
        ],
    context={
        'sph-lead': 0
    }
)

print("Initializing visualization...")
fig, ax = viz.set_up_figure(title="Benchmark Mission 1 with Smoothed Particle Hydrodynamics")

print("Starting sim...")

for i in range(10000):

    world.advance_state()

    if i % 50 == 0:
        viz.render_2d_orbit_state(
            world.get_state(),
            fig,
            ax,
            agent_colors=['k']*4+['b'],
            agent_sizes=[100]*5
        )