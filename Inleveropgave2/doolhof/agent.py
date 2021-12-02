from Inleveropgave2.doolhof.policy import Policy
from Inleveropgave2.doolhof.doolhof import Doolhof
from Inleveropgave2.doolhof.state import State
import copy
import random


class Agent:
    def __init__(self, doolhof: Doolhof, policy: Policy, startState: State):
        self.doolhof = doolhof
        self.policy = policy
        self.state = startState

    def bellmanEquation(self, location: tuple, action: int, discount: float):
        """Calculates the bellman equation (reward + discount * value)"""
        pos = self.doolhof.action[action](location)  # if I execute the action what will the position be
        index = self.doolhof.coordsToIndex(pos)  # position to index
        if not self.doolhof.canIGoThere(index):  # check if the new position is out of the map
            index = self.doolhof.coordsToIndex(location)
        value = self.doolhof.map[index[0]][index[1]].value
        reward = self.doolhof.map[index[0]][index[1]].reward
        return reward + discount * value

    def valueCalculate(self, location: tuple, discount: float = 1, deterministic: bool = True):
        """Calculates the value for the valuefunction of one location in the map"""
        values = []
        actions = self.doolhof.action.keys()
        for action in actions:
            if not deterministic:
                sum = 0
                for possibleNextState in actions:
                    value = self.bellmanEquation(location, possibleNextState, discount)
                    sum += (self.doolhof.actionConsequenceChance[possibleNextState] * value)
                values.append(sum)
            else:
                value = self.bellmanEquation(location, action, discount)
                values.append(value)
        return max(values)

    def choseAction(self, discount):
        """This function choses an action with the policy"""
        actions = self.policy.selectAction(pos=self.state.position, discount=discount)
        return actions[0]

    def valueIteration(self, discount: float, threshhold=0.01, deterministic: bool = True):
        """Makes the valuefunction. Loops thru the whole map and calculates the values"""
        newValues = copy.deepcopy(self.doolhof.map)
        done = False
        iteration = 0
        while not done:
            delta = 0
            height = len(self.doolhof.map)
            width = len(self.doolhof.map[0])
            for y in range(height):  # Loop thru the whole map
                for x in range(width):
                    index = self.doolhof.coordsToIndex((x, y))
                    value = self.valueCalculate(location=(x, y), discount=discount, deterministic=deterministic)
                    if self.doolhof.map[index[0]][index[1]].done:
                        value = 0
                    newValues[index[0]][index[1]].value = value  # set new value
                    #  Calculate the current delta
                    oldValue = self.doolhof.map[index[0]][index[1]].value
                    delta = max(delta, abs(oldValue - value))
            self.doolhof.map = copy.deepcopy(newValues)
            if delta < threshhold:
                done = True
            iteration += 1
        return newValues

    def createRandomRoute(self):
        pos = (random.randint(0, 3), random.randint(0, 3))  # start position
        route = [pos]
        index = self.doolhof.coordsToIndex(pos)
        for i in range(1000):
            chosenAction = random.choice(self.policy.matrix[index[0]][index[1]])
            if chosenAction == None:
                break
            action = self.doolhof.action[chosenAction]
            newPos = action(pos)
            index = self.doolhof.coordsToIndex(newPos)
            if not self.doolhof.canIGoThere(index):
                newPos = copy.deepcopy(pos)
                index = self.doolhof.coordsToIndex(newPos)
            route.append(newPos)
            pos = copy.deepcopy(newPos)
        return route

    def monteCarloPolicyEvaluation(self, episodes: int = 1000, discount: float = 0.9):
        for episode in range(episodes):
            route = self.createRandomRoute()
            returns = [0]
            index = self.doolhof.coordsToIndex(route[0])
            self.state = self.doolhof.map[index[0]][index[1]]
            for i in range(len(route) - 1):
                if self.state.done:
                    break
                i += 1
                index = self.doolhof.coordsToIndex(route[i])
                self.state = self.doolhof.map[index[0]][index[1]]
                return_ = discount * returns[-1] + self.state.reward
                returns.append(return_)
            route = route[:len(returns)]
            routesDone = []
            for i in range(len(route)):
                i = (i + 1) * - 1
                pos = route[i]
                if not pos in routesDone:
                    routesDone.append(pos)
                    index = self.doolhof.coordsToIndex(pos)
                    state = self.doolhof.map[index[0]][index[1]]
                    if state.done:
                        state.value = 0
                    else:
                        state.value = (state.value + returns[i]) / 2
        print(self.doolhof.getValues())

    def temporalDifferenceLearning(self, episodes: int = 1, discount: float = 0.9):
        for episode in range(episodes):
            pos = (random.randint(0, 3), random.randint(0, 3))
            route = [pos]
            returns = [0]
            index = self.doolhof.coordsToIndex(pos)
            self.state = self.doolhof.map[index[0]][index[1]]
            while self.state.done == False:
                if self.state.done:
                    break
                actions = self.policy.matrix[index[0]][index[1]]
                action = random.choice(actions)
                newPos = self.doolhof.action[action](pos)
                if not self.doolhof.canIGoThere(newPos):
                    newPos = copy.deepcopy(pos)
                pos = copy.deepcopy(newPos)
                route.append(newPos)
                index = self.doolhof.coordsToIndex(newPos)
                self.state = self.doolhof.map[index[0]][index[1]]
                return_ = discount * returns[-1] + self.state.reward
                returns.append(return_)
            routesDone = []
            for i in range(len(route)):
                i = (i + 1) * - 1
                pos = route[i]
                if not pos in routesDone:
                    routesDone.append(pos)
                    index = self.doolhof.coordsToIndex(pos)
                    state = self.doolhof.map[index[0]][index[1]]
                    state.value = (state.value + returns[i]) / 2
        return self.doolhof.getValues()

    def __str__(self):
        return "state: %s" % self.state
