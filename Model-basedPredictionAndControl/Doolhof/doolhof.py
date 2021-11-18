from state import state


class doolhof:
    def __init__(self, state: state):
        self.map = [[0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]]
        self.rewards = [[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]]
        self.state = state
        self.action = {0: (lambda q: (q[0], q[1] + 1)),
                       1: (lambda q: (q[0] + 1, q[1])),
                       2: (lambda q: (q[0], q[1] - 1)),
                       3: (lambda q: (q[0] - 1, q[1]))}

    def step(self, state: state, action: int) -> state:
        pass

    def __str__(self):
        return "map: %s \n rewards: %s \n state: %s" % (self.map, self.rewards, self.state)
