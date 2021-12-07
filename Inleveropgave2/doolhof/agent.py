from Inleveropgave2.doolhof.policy import Policy, getMaxFromList
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

    def createRouteByPolicy(self):
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
        returns = {}
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                returns[(x, y)] = []
        for episode in range(episodes):
            route = self.createRouteByPolicy()
            g = 0
            for i in range(len(route) - 2, -1, -1):
                if not route[i] in route[:i]:
                    index = self.doolhof.coordsToIndex(route[i])
                    state = self.doolhof.map[index[0]][index[1]]
                    nextIndex = self.doolhof.coordsToIndex(route[i + 1])
                    nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                    g = discount * g + nextState.reward
                    returns[route[i]].append(g)
                    if state.done:
                        state.value = 0
                    else:
                        state.value = sum(returns[route[i]]) / len(returns[route[i]])
        return self.doolhof.getValues()

    def tabular(self, episodes: int = 1, discount: float = 1, alpha: float = 0.1):
        for episode in range(episodes):
            pos = (random.randint(0, 3), random.randint(0, 3))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            while not state.done:
                action = random.choice(self.policy.matrix[index[0]][index[1]])
                nextPos = self.doolhof.action[action](pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                if not self.doolhof.canIGoThere(nextIndex):
                    nextPos = copy.deepcopy(pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                state.value = state.value + alpha * (nextState.reward + discount * nextState.value - state.value)
                state = nextState
                pos = copy.deepcopy(nextPos)
                index = copy.deepcopy(nextIndex)
        return self.doolhof.getValues()




    def createRouteWithActionByPolicy(self):
        pos = (random.randrange(4), random.randrange(4))  # start position
        route = [(pos, 0)]
        index = self.doolhof.coordsToIndex(pos)
        while not self.doolhof.map[index[0]][index[1]].done:
            currPolicy = self.policy.matrix3D[index[0]][index[1]]
            # chosenAction = getMaxAction(currPolicy)
            chosenAction = random.choices(list(self.doolhof.action.keys()), currPolicy)[0]
            action = self.doolhof.action[chosenAction]
            newPos = action(pos)
            index = self.doolhof.coordsToIndex(newPos)
            if not self.doolhof.canIGoThere(index):
                newPos = copy.deepcopy(pos)
                index = self.doolhof.coordsToIndex(newPos)
            route.append((newPos, chosenAction))
            pos = copy.deepcopy(newPos)
        return route

    def onPolicyFirstVisitMonteCarloControl(self, episodes: int = 1, discount: float = 0.9, epsilon=0.1):
        actions = sorted(list(self.doolhof.action.keys()))
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        returns = {}
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                for action in actions:
                    returns[(x, y), action] = []
        for episode in range(episodes):
            # print("episode", episode)
            # print(self.policy.matrix3D)
            route = self.createRouteWithActionByPolicy()
            len_ = len(route)
            g = 0
            for i in range(len(route) - 2, -1, -1):
                index = self.doolhof.coordsToIndex(route[i][0])
                state = self.doolhof.map[index[0]][index[1]]
                nextIndex = self.doolhof.coordsToIndex(route[i + 1][0])
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                g = discount * g + nextState.reward

                if not route[i] in route[:i]:
                    returns[(route[i])].append(g)

                    q[index[0]][index[1]][route[i][1]] = sum(returns[route[i]]) / len(returns[route[i]])

                    aStar = getMaxFromList(q[index[0]][index[1]])

                    for a in actions:
                        if a == aStar:
                            policyA = 1 - epsilon + epsilon / len(q[index[0]][index[1]])
                        else:
                            policyA = epsilon / len(q[index[0]][index[1]])
                        self.policy.matrix3D[index[0]][index[1]][a] = policyA

        return q

    def sarsa(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha = 0.1):
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        for episode in range(episodes):
            pos = (random.randrange(4), random.randrange(4))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            action = random.choices(sorted(list(self.doolhof.action.keys())), q[index[0]][index[1]])[0]
            while not state.done:
                nextPos = self.doolhof.action[action](pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                if not self.doolhof.canIGoThere(nextIndex):
                    nextPos = copy.deepcopy(pos)
                    nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                reward = nextState.reward
                # print(self.policy.matrix3D[nextIndex[0]][nextIndex[1]])
                nextAction = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix3D[nextIndex[0]][nextIndex[1]])[0]
                # print("nextAction", nextAction)

                currQ = q[index[0]][index[1]][action]
                nextQ = q[nextIndex[0]][nextIndex[1]][nextAction]
                q[index[0]][index[1]][action] = currQ + alpha * (reward + discount * nextQ - currQ)
                self.policy.updatePolicyMatrix(q=q, index=index, epsilon=epsilon)

                pos = copy.deepcopy(nextPos)
                index = copy.deepcopy(nextIndex)
                state = copy.deepcopy(nextState)
                action = copy.deepcopy(nextAction)
            print(episode)
        return q

    def qLearning(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1):
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        for episode in range(episodes):
            pos = (random.randint(0, 3), random.randint(0, 3))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            action = q[index[0]][index[1]].index(max(q[index[0]][index[1]]))
            while not state.done:
                nextPos = self.doolhof.action[action](pos)
                if not self.doolhof.canIGoThere(nextPos):
                    nextPos = pos
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                reward = nextState.reward
                # q[index[0]][index[1]][action] = q[index[0]][index[1]][action] + epsilon * (reward + discount * # ?????)


    def __str__(self):
        return "state: %s" % self.state
