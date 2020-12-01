

class Agent:

    next_id = 0

    def __init__(self):

        self.id = Agent.next_id
        Agent.next_id += 1



class Mothership (Agent):
    pass
