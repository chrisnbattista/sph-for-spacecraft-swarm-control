





from sfc.multi_agent_kinetics import experiments, viz, worlds, forces

print("Initiating benchmark mission...")

print("Seeding initial state...")
world = worlds.World(
    n_agents=5,
    forces=[lambda x: forces.gravity_well(x, 1)]
)
world.set_state(experiments.set_up_experiment(5, 20))

print("Initializing visualization...")
fig, ax = viz.set_up_figure()

for i in range(100):

    world.time_step()

    viz.render_state(
        world.get_state(),
        fig,
        ax
    )