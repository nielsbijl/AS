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

    def step(self, state: state, action: int) -> state:
        if action == 0:  # move up
            newPos = (state.position[0], state.position[1] + 1)
            newSate = state(newPos, )
            pass
        elif action == 1:  # move right
            pass
        elif action == 2:  # move down
            pass
        else:  # move left
            pass

    def __str__(self):
        return "map: %s \n rewards: %s \n state: %s" % (self.map, self.rewards, self.state)
