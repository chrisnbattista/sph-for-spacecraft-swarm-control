



class World:
    '''
    Stores the world state and provides convenience functions to update agents.
    '''

    def __init__(self, agents=[]):
        self.agent_list = agents

    def update(self, time):
        for a in self.agent_list:
            a.update(self, time)
