from ModelbasedPredictionAndControl.doolhof.state import state


class doolhof:
    def __init__(self):
        self.action = {0: (lambda q: (q[0], q[1] + 1)),
                       1: (lambda q: (q[0] + 1, q[1])),
                       2: (lambda q: (q[0], q[1] - 1)),
                       3: (lambda q: (q[0] - 1, q[1]))}
        self.actionConsequenceChance = {0: 0.7,
                                        1: 0.1,
                                        2: 0.1,
                                        3: 0.1}
        self.map = [[state((0, 3), -1, False), state((1, 3), -1, False), state((2, 3), -1, False), state((3, 3), 40, True)],
                    [state((0, 2), -1, False), state((1, 2), -1, False), state((2, 2), -10, False), state((3, 2), -10, False)],
                    [state((0, 1), -1, False), state((1, 1), -1, False), state((2, 1), -1, False), state((3, 1), -1, False)],
                    [state((0, 0), 10, True), state((1, 0), -2, False), state((2, 0), -1, False), state((3, 0), -1, False)]]

    def step(self, state: state, action: int) -> state:
        newPos = self.action[action](state.position)
        indexNewPos = self.coordsToIndex(newPos)
        if (indexNewPos[0] < 0 or indexNewPos[1] < 0) or (
                indexNewPos[0] > (len(self.map) - 1) or indexNewPos[1] > (len(self.map) - 1)):
            newPos = state.position
            indexNewPos = self.coordsToIndex(newPos)
        reward = self.map[indexNewPos[0]][indexNewPos[1]].reward
        state.position = newPos
        state.reward += reward
        if self.map[indexNewPos[0]][indexNewPos[1]].done:
            state.done = True
        else:
            state.done = False
        return state

    def run(self, agent, deterministic: bool = True, discount: float = 0.9):
        path = [agent.state.position]
        while not agent.state.done:
            agent.valueIteration(discount=discount, threshhold=0.01, deterministic=deterministic)
            action = agent.choseAction(discount=discount)
            agent.state = self.step(state=agent.state, action=action)
            path.append(agent.state.position)
        return path

    def coordsToIndex(self, pos: tuple):
        return len(self.map) - pos[1] - 1, pos[0]

    def indexToCoords(self, index: tuple):
        return index[1], len(self.map) - index[0] - 1

    def canIGoThere(self, index: tuple):
        if (index[0] < 0 or index[1] < 0) or (
                index[0] > (len(self.map) - 1) or index[1] > (len(self.map) - 1)):
            return False
        return True

    def getValues(self):
        return [[i.value for i in row] for row in self.map]

    def getRewards(self):
        return [[i.reward for i in row] for row in self.map]

    def __str__(self):
        return "values: %s \n rewards: %s \n" % (self.getValues(), self.getRewards())
