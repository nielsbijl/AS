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


    # def createRouteWithActionByPolicy(self):
    #     # pos = (random.randrange(4), random.randrange(4))  # start position
    #     pos = (2, 0)
    #     action = random.randrange(4)
    #     route = []
    #     index = self.doolhof.coordsToIndex(pos)
    #     state = self.doolhof.map[index[0]][index[1]]
    #     while not state.done:
    #         route.append((pos, action))
    #         currPolicy = self.policy.matrix3D[index[0]][index[1]]
    #         action = copy.deepcopy(random.choices(sorted(list(self.doolhof.action.keys())), currPolicy)[0])
    #         newPos = self.doolhof.action[action](pos)
    #         newIndex = self.doolhof.coordsToIndex(newPos)
    #         if not self.doolhof.canIGoThere(newIndex):
    #             newPos = copy.deepcopy(pos)
    #             newIndex = self.doolhof.coordsToIndex(newPos)
    #         state = copy.deepcopy(self.doolhof.map[newIndex[0]][newIndex[1]])
    #         pos = copy.deepcopy(newPos)
    #     route.append((pos, action))
    #     return route

    def createEpisodeRoute(self):
        route = []
        pos = (random.randrange(4), random.randrange(4))  # start position
        index = self.doolhof.coordsToIndex(pos)
        state = self.doolhof.map[index[0]][index[1]]
        action = random.randrange(4)
        while not state.done:
            action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix3D[index[0]][index[1]])[0]
            route.append((pos, action))
            newPos = self.doolhof.action[action](pos)
            if not self.doolhof.canIGoThere(self.doolhof.coordsToIndex(newPos)):
                newPos = copy.deepcopy(pos)
            pos = copy.deepcopy(newPos)
            index = self.doolhof.coordsToIndex(newPos)
            state = self.doolhof.map[index[0]][index[1]]
        route.append((pos, action))
        return route



    def onPolicyFirstVisitMonteCarloControl(self, episodes: int = 100000, discount: float = 0.9, epsilon=0.1):
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        returns = {}
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                for action in list(self.doolhof.action.keys()):
                    returns[(x, y), action] = []
        for episode in range(episodes):
            g = 0
            episodeRoute = self.createEpisodeRoute()
            for t in range(-1, len(episodeRoute) * - 1, - 1):
                pos = episodeRoute[t - 1][0]
                index = self.doolhof.coordsToIndex(pos)
                action = episodeRoute[t][1]
                nextPos = episodeRoute[t][0]
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                reward = self.doolhof.map[nextIndex[0]][nextIndex[1]].reward
                g = discount * g + reward
                if not (episodeRoute[t - 1] in episodeRoute[:t - 1]):
                    returns[(pos, action)].append(g)
                    q[index[0]][index[1]][action] = sum(returns[(pos, action)]) / len(returns[(pos, action)])
                    self.policy.updatePolicyMatrix(q=q, index=index, epsilon=epsilon)
            print(episode)
        return q





    # def onPolicyFirstVisitMonteCarloControl(self, episodes: int = 100000, discount: float = 0.9, epsilon=0.1):
    #     actions = sorted(list(self.doolhof.action.keys()))
    #     q = [[[0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
    #          [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    #          [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
    #          [[0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]]
    #     returns = {}
    #     for y in range(len(self.doolhof.map)):
    #         for x in range(len(self.doolhof.map[0])):
    #             for action in actions:
    #                 returns[(x, y), action] = []
    #     for episode in range(episodes):
    #         # print("episode", episode)
    #         # print(self.policy.matrix3D)
    #         route = self.createRouteWithActionByPolicy()
    #         len_ = len(route)
    #         print('route len: ', len_)
    #         g = 0
    #         for i in range(-1, len(route) * - 1, -1):
    #             pos = route[i][0]
    #             index = self.doolhof.coordsToIndex(pos)
    #             state = self.doolhof.map[index[0]][index[1]]
    #             nextPos = route[i - 1][0]
    #             nextIndex = self.doolhof.coordsToIndex(nextPos)
    #             nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
    #             g = discount * g + nextState.reward
    #
    #             if not route[i] in route[:i]:
    #                 returns[(route[i])].append(g)
    #
    #                 q[index[0]][index[1]][route[i][1]] = sum(returns[route[i]]) / len(returns[route[i]])
    #
    #                 self.policy.updatePolicyMatrix(q=q, index=index, epsilon=epsilon)
    #         print(episode)
    #     return q

    def sarsa(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha = 0.1):
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        for episode in range(episodes):
            pos = (random.randrange(4), random.randrange(4))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix3D[index[0]][index[1]])[0]
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

    def qLearning(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha = 0.1):
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        for episode in range(episodes):
            pos = (random.randint(0, 3), random.randint(0, 3))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            while not state.done:
                action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix3D[index[0]][index[1]])[0]
                nextPos = self.doolhof.action[action](pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                if not self.doolhof.canIGoThere(nextPos):
                    nextPos = pos
                    nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                reward = nextState.reward
                a = getMaxFromList(q[nextIndex[0]][nextIndex[1]])
                q[index[0]][index[1]][action] = q[index[0]][index[1]][action] + alpha * (reward + discount * q[nextIndex[0]][nextIndex[1]][a] - q[index[0]][index[1]][action])
                self.policy.updatePolicyMatrix(q, index, epsilon)
                state = copy.deepcopy(nextState)
                index = copy.deepcopy(nextIndex)
                pos = copy.deepcopy(nextPos)
            print(episode)

        return q

    def sumQs(self, q1, q2):
        q = []
        for i in range(len(q1)):
            row = []
            row1 = q1[i]
            row2 = q2[i]
            for k in range(len(row1)):
                list_ = []
                list1 = row1[k]
                list2 = row2[k]
                for v in range(len(list1)):
                    list_.append(list1[v] + list2[v])
                row.append(list_)
            q.append(row)
        return q


    def doubleqLearning(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha = 0.1):
        q1 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        q2 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        for episode in range(episodes):
            pos = (random.randint(0, 3), random.randint(0, 3))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            while not state.done:
                action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix3D[index[0]][index[1]])[0]
                nextPos = self.doolhof.action[action](pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                if not self.doolhof.canIGoThere(nextPos):
                    nextPos = pos
                    nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                reward = nextState.reward

                if random.randint(0,1) > 0:
                    a = getMaxFromList(q1[nextIndex[0]][nextIndex[1]])
                    q1[index[0]][index[1]][action] = q1[index[0]][index[1]][action] + alpha * (reward + discount * q2[nextIndex[0]][nextIndex[1]][a] - q1[index[0]][index[1]][action])
                else:
                    a = getMaxFromList(q2[nextIndex[0]][nextIndex[1]])
                    q2[index[0]][index[1]][action] = q2[index[0]][index[1]][action] + alpha * (reward + discount * q1[nextIndex[0]][nextIndex[1]][a] - q2[index[0]][index[1]][action])
                q = self.sumQs(q1, q2)
                self.policy.updatePolicyMatrix(q, index, epsilon)
                state = copy.deepcopy(nextState)
                index = copy.deepcopy(nextIndex)
                pos = copy.deepcopy(nextPos)
            print(episode)

        return q

    def __str__(self):
        return "state: %s" % self.state
