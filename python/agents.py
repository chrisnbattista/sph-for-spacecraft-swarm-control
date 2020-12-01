

class Agent:

    next_id = 0

    def __init__(self, x=0, y=0, u=0, v=0, m=1, visc=1, bulk_mod=1):

        self.id = Agent.next_id
        Agent.next_id += 1

        self.state = {
            'x': x,
            'y': y,
            'u': u,
            'v': v
        }

        self.props = {
            'm': m,
            'visc': visc,
            'bulk_mod': bulk_mod
        }

    def update(self, world):
        '''
        Updates agent internal state based on world snapshot.
        Returns dictionary of state updates to be applied to world snapshot.
        '''
        return {}


class Mothership (Agent):
    pass
