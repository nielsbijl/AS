from ModelbasedPredictionAndControl.doolhof.state import state
import random


class doolhof:
    def __init__(self, state: state):
        self.values = [[0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0],
                       [0, 0, 0, 0]]
        self.rewards = [[-1, -1, -1, 40],
                        [-1, -1, -10, -10],
                        [-1, -1, -1, -1],
                        [10, -2, -1, -1]]
        self.end = [[False, False, False, True],
                    [False, False, False, False],
                    [False, False, False, False],
                    [True, False, False, False]]
        self.state = state
        self.action = {0: (lambda q: (q[0], q[1] + 1)),
                       1: (lambda q: (q[0] + 1, q[1])),
                       2: (lambda q: (q[0], q[1] - 1)),
                       3: (lambda q: (q[0] - 1, q[1]))}
        self.actionChance = {0: 0.7,
                             1: 0.1,
                             2: 0.1,
                             3: 0.1}

    def step(self, state: state, action: int) -> state:
        newPos = self.action[action](state.position)
        indexNewPos = self.coordsToIndex(newPos)
        if (indexNewPos[0] < 0 or indexNewPos[1] < 0) or (
                indexNewPos[0] > (len(self.values) - 1) or indexNewPos[1] > (len(self.values) - 1)):
            newPos = state.position
            indexNewPos = self.coordsToIndex(newPos)
        reward = self.rewards[indexNewPos[0]][indexNewPos[1]]
        state.position = newPos
        state.reward += reward
        if self.end[indexNewPos[0]][indexNewPos[1]]:
            state.done = True
        else:
            state.done = False
        return state

    def run(self, agent):
        print(self.state.position)
        while not self.state.done:
            agent.valueIteration(discount=0.9, threshhold=0.01)
            actions = agent.policy.selectAction(pos=self.state.position, discount=0.9)
            action = random.choice(actions)
            self.step(state=self.state, action=action)
            print(self.state.position)

    def coordsToIndex(self, pos: tuple):
        return len(self.rewards) - pos[1] - 1, pos[0]

    def indexToCoords(self, index: tuple):
        return index[1], len(self.rewards) - index[0] - 1

    def __str__(self):
        return "map: %s \n rewards: %s \n state: %s" % (self.values, self.rewards, self.state)
