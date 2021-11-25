from ModelbasedPredictionAndControl.doolhof.state import State


class Doolhof:
    def __init__(self):
        self.action = {0: (lambda q: (q[0], q[1] + 1)),  # Go up
                       1: (lambda q: (q[0] + 1, q[1])),  # Go right
                       2: (lambda q: (q[0], q[1] - 1)),  # Go down
                       3: (lambda q: (q[0] - 1, q[1]))}  # Go left
        self.actionConsequenceChance = {0: 0.7,
                                        1: 0.1,
                                        2: 0.1,
                                        3: 0.1}
        # The map is a 2-d list of states
        self.map = [[State((0, 3), -1, False), State((1, 3), -1, False), State((2, 3), -1, False), State((3, 3), 40, True)],
                    [State((0, 2), -1, False), State((1, 2), -1, False), State((2, 2), -10, False), State((3, 2), -10, False)],
                    [State((0, 1), -1, False), State((1, 1), -1, False), State((2, 1), -1, False), State((3, 1), -1, False)],
                    [State((0, 0), 10, True), State((1, 0), -2, False), State((2, 0), -1, False), State((3, 0), -1, False)]]

    def step(self, currentState: State, action: int) -> State:
        """This function executes one step of the simulation"""
        newPos = self.action[action](currentState.position)
        indexNewPos = self.coordsToIndex(newPos)
        if not self.canIGoThere(indexNewPos):  # if its out of the map he takes the current position
            newPos = currentState.position
            indexNewPos = self.coordsToIndex(newPos)
        return self.map[indexNewPos[0]][indexNewPos[1]]

    def run(self, agent, deterministic: bool = True, discount: float = 0.9):
        """This function runs the simulation. It let's the agent walk thru the map. It returns the path the agent has
        traveled"""
        path = [agent.state.position]
        while not agent.state.done:
            agent.valueIteration(discount=discount, threshhold=0.01, deterministic=deterministic)
            action = agent.choseAction(discount=discount)
            agent.state = self.step(currentState=agent.state, action=action)
            path.append(agent.state.position)
        return path

    def coordsToIndex(self, pos: tuple):
        """Converts coordinates to the map index"""
        return len(self.map) - pos[1] - 1, pos[0]

    def indexToCoords(self, index: tuple):
        """Converts the map index to coordinates"""
        return index[1], len(self.map) - index[0] - 1

    def canIGoThere(self, index: tuple):
        """Checks if this position is inside the map"""
        if (index[0] < 0 or index[1] < 0) or (
                index[0] > (len(self.map) - 1) or index[1] > (len(self.map) - 1)):
            return False
        return True

    def getValues(self):
        """"Function to easily get all the current values"""
        return [[i.value for i in row] for row in self.map]

    def getRewards(self):
        """"Function to easily get all the current rewards"""
        return [[i.reward for i in row] for row in self.map]

    def __str__(self):
        return "values: %s \n rewards: %s \n" % (self.getValues(), self.getRewards())
