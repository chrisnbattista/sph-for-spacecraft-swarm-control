





import numpy as np






class World:
    '''
    Stores the world state and provides convenience functions to update agents.
    Represents a shared snapshot of the environment and agent state to
    facilitate cooperative distributed computing.
    '''

    def __init__(self, agent_list=[], t=0):
        self.agent_list = agent_list
        self.state = {
            't': t,
            'positions': self.get_positions_dict()
        }


    def get_positions_dict(self):
        '''
        Return a dictionary with all agent positions.
        '''

        return {a.id: (a.state['x'], a.state['y']) for a in self.agent_list}


    def get_positions_array(self):
        '''
        Return a NumPy dictionary with all agent positions.
        '''

        self.agent_list.sort(key=lambda x: x.id)
        return np.array([
            np.vectorize(lambda x: x.state['x'])(self.agent_list),
            np.vectorize(lambda x: x.state['y'])(self.agent_list)
        ]).T


    def update(self, time):
        '''
        Provides a utility to update agent states in a "simultaneous" fashion
        and step the shared world state representation forward once all agent-
        level updates are complete.
        '''

        # Set shared timestamp
        self.state['t'] = time

        # Prepare to store state updates from agents
        state_updates = []

        # Update each agent in turn, but cache their state updates
        for a in self.agent_list:
            state_updates.append(
                a.update(self)
            )

        for u in state_updates:
            # Recursively cascade state updates (diffs) down the tree of state
            # Only goes one level deep with recursive update
            for key in u:
                if type(u[key]) == dict:
                    self.state[key].update(u[key])
                else:
                    self.state[key] = u[key]
