from ModelbasedPredictionAndControl.doolhof.state import state


class doolhof:
    def __init__(self, state: state, rewards: list):
        self.values = [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
        self.rewards = rewards
        self.state = state
        self.action = {0: (lambda q: (q[0], q[1] + 1)),
                       1: (lambda q: (q[0] + 1, q[1])),
                       2: (lambda q: (q[0], q[1] - 1)),
                       3: (lambda q: (q[0] - 1, q[1]))}

    def step(self, state: state, action: int) -> state:

        pass

    def __str__(self):
        return "map: %s \n rewards: %s \n state: %s" % (self.values, self.rewards, self.state)
