





import kernels, viz
from agents import Agent, Mothership
from worlds import World

import configparser, random


if __name__ == '__main__':

    ## Configuration
    # Settings are outsourced to a standard ini file
    config = configparser.ConfigParser()
    config.read('config.ini')

    ## Initialization
    # Using expected setup with four "daughter ships" and one "mother ship"
    agent_list = [
                    Agent(
                        x=(random.random()-0.5) * config['init'].getfloat('spread') * 2,
                        y=(random.random()-0.5) * config['init'].getfloat('spread') * 2
                        ) \
                for i in range(4)] \
                + [
                    Mothership(
                        x=(random.random()-0.5) * config['init'].getfloat('spread') * 2,
                        y=(random.random()-0.5) * config['init'].getfloat('spread') * 2
                    )
                ]
    world = World(agent_list)

    viz.init()

    ## Event loop
    # This is where the simulation is done
    # This is also where the graphical display is updated
    for i in range(config['sim'].getint('n_timesteps')):

        world.update(i * config['sim'].getfloat('timestep'))

        viz.draw(world)
