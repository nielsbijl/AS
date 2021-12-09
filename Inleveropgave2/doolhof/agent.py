from Inleveropgave2.doolhof.policy import Policy, getMaxFromList
from Inleveropgave2.doolhof.maze import Maze
from Inleveropgave2.doolhof.state import State
import copy
import random


class Agent:
    def __init__(self, doolhof: Maze, policy: Policy, startState: State):
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

    def monteCarloPolicyEvaluation(self, episodes: int = 1000, discount: float = 0.9):
        """Makes the valuefunction. First-visist MC prediction"""
        # Initialize Returns(S)
        returns = {}
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                returns[(x, y)] = []
        for episode in range(episodes):  # loop the amount of episodes
            # Generate an episode (route) following policy
            route = [item[0] for item in self.createEpisodeRoute()]  # [(x, y), (x, y), (x, y).....]
            # G <- 0
            g = 0
            # Walk backwards through the route
            for t in range(-1, len(route) * - 1, - 1):
                # G <- discount * G + Reward
                index = self.doolhof.coordsToIndex(route[t - 1])  # Set coordinates to index
                state = self.doolhof.map[index[0]][index[1]]  # Get the state from the maze map
                nextIndex = self.doolhof.coordsToIndex(route[t])  # Set next coordinates to the next index
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]  # Get the next state from the maze map
                g = discount * g + nextState.reward
                # Unless St appears in S0, S1, ....
                if not route[t-1] in route[:t-1]:
                    # Append G to Returns(St)
                    returns[route[t-1]].append(g)
                    # V(St) <- average(Returns(St))
                    state.value = sum(returns[route[t-1]]) / len(returns[route[t-1]])
        return self.doolhof.getValues()  # Return valuefunction

    def tabular(self, episodes: int = 1, discount: float = 1, alpha: float = 0.1):
        """Makes the valuefunction. Temporal Difference Learning"""
        for episode in range(episodes):
            # Initialize S
            pos = (random.randint(0, 3), random.randint(0, 3))
            index = self.doolhof.coordsToIndex(pos)
            state = self.doolhof.map[index[0]][index[1]]
            while not state.done:  # Loop for each step of episode until S is terminal
                # Action given by policy for S
                action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix[index[0]][index[1]])[0]
                # Take action A, observe R, S'
                nextPos = self.doolhof.action[action](pos)
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                if not self.doolhof.canIGoThere(nextIndex):  # check if next pos is out of the map
                    nextPos = copy.deepcopy(pos)
                    nextIndex = self.doolhof.coordsToIndex(nextPos)
                nextState = self.doolhof.map[nextIndex[0]][nextIndex[1]]
                # V(S) <- V(S) + alpha * [R + discount * V(S') - V(S)]
                state.value = state.value + alpha * (nextState.reward + discount * nextState.value - state.value)
                # S <- S'
                state = copy.deepcopy(nextState)
                pos = copy.deepcopy(nextPos)
                index = copy.deepcopy(nextIndex)
        return self.doolhof.getValues()  # Return valuefunction

    def createEpisodeRoute(self):
        """Creates an episode/route, list of positions and actions"""
        # Initialize empty episode (route)
        route = []  # format: [((x, y), action), ((x, y), action) .....]
        # Initialize start state
        self.state = self.doolhof.map[random.randrange(4)][random.randrange(4)]
        state = self.state
        pos = state.position
        # Initialize start action
        while not state.done:
            # Chose action following policy
            index = self.doolhof.coordsToIndex(pos)
            action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix[index[0]][index[1]])[
                0]
            route.append((pos, action))
            # Set next State
            state = self.doolhof.step(currentState=state, action=action)
            pos = state.position
        route.append((pos, None))
        return route

    def onPolicyFirstVisitMonteCarloControl(self, episodes: int = 100000, discount: float = 0.9, epsilon=0.1):
        """The on policy first visit monte carlo control algorithm
        :returns Q function
        """
        # Initialize Q(S,A)
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # Initialize Returns(s, a)
        returns = {}
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                for action in list(self.doolhof.action.keys()):
                    returns[(x, y), action] = []
        # Repeat forever (for each episode)
        for episode in range(episodes):
            # Generate an episode (route) following policy
            episodeRoute = self.createEpisodeRoute()
            # G <- 0
            g = 0
            # Loop for each step of episode
            for t in range(-1, len(episodeRoute) * - 1, - 1):  # Walk backwards through the route
                # G <- discount * G + R
                pos = episodeRoute[t - 1][0]
                index = self.doolhof.coordsToIndex(pos)
                action = episodeRoute[t - 1][1]
                nextPos = episodeRoute[t][0]
                nextIndex = self.doolhof.coordsToIndex(nextPos)
                reward = self.doolhof.map[nextIndex[0]][nextIndex[1]].reward
                g = discount * g + reward
                # Unless the pair St,At appears in S0,A0,S1,A1 .....

                if not (episodeRoute[t - 1] in episodeRoute[:t - 1]):
                    # Append G to Returns(St,At)
                    returns[(pos, action)].append(g)
                    # Q(St,At) <- average(Returns(St,At))
                    q[index[0]][index[1]][action] = sum(returns[(pos, action)]) / len(returns[(pos, action)])
                    # Update policy with Q
                    self.policy.updatePolicyMatrix(q=q, index=index, epsilon=epsilon)
        return q

    def sarsa(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha=0.1):
        """The Sarsa alorithm
        :returns Q function
        """
        # Initialize Q(s, a)
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # Loop for each episode
        for episode in range(episodes):
            # Initialize S
            self.state = self.doolhof.map[random.randrange(4)][random.randrange(4)]
            pos = self.state.position
            index = self.doolhof.coordsToIndex(pos)
            # Choose A from S using policy derived from Q
            action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix[index[0]][index[1]])[
                0]
            # Loop for each step of episode until S is terminal
            while not self.state.done:
                # Take action A, observe S'
                nextState = self.doolhof.step(currentState=self.state, action=action)
                nextIndex = self.doolhof.coordsToIndex(nextState.position)
                # Take action A, observe R
                reward = nextState.reward
                # Choose A' from S' using policy derived from Q
                nextAction = random.choices(sorted(list(self.doolhof.action.keys())),
                                            self.policy.matrix[nextIndex[0]][nextIndex[1]])[0]
                # Q(S, A) <- Q(S, A) + alpha[R + discound * Q(S', A') - Q(S, A)]
                currQ = q[index[0]][index[1]][action]
                nextQ = q[nextIndex[0]][nextIndex[1]][nextAction]
                q[index[0]][index[1]][action] = currQ + alpha * (reward + discount * nextQ - currQ)

                # Update policy derived from Q
                self.policy.updatePolicyMatrix(q=q, index=index, epsilon=epsilon)

                # S <- S'
                index = copy.deepcopy(nextIndex)
                self.state = copy.deepcopy(nextState)
                # A <- A'
                action = copy.deepcopy(nextAction)
        return q

    def qLearning(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha=0.1):
        """Q-learning algorithm
        :returns Q-functions"""
        # Initialize Q(s, a)
        q = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
             [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # Loop for each episode
        for episode in range(episodes):
            # Initialize S
            self.state = self.doolhof.map[random.randrange(4)][random.randrange(4)]
            index = self.doolhof.coordsToIndex(self.state.position)
            # Loop for each step of episode until S is terminal
            while not self.state.done:
                # Choose A from S using policy derived from Q
                action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix[index[0]][index[1]])[0]
                # Take action A, observe S'
                nextState = self.doolhof.step(currentState=self.state, action=action)
                nextIndex = self.doolhof.coordsToIndex(nextState.position)
                # Take action A, observe R
                reward = nextState.reward
                # Q(S, A) <- Q(S, A) + alpha[R + discound * MAXa(Q(S', A)) - Q(S, A)]
                a = getMaxFromList(q[nextIndex[0]][nextIndex[1]])
                q[index[0]][index[1]][action] = q[index[0]][index[1]][action] + alpha * (
                            reward + discount * q[nextIndex[0]][nextIndex[1]][a] - q[index[0]][index[1]][action])
                # Update policy derived from Q
                self.policy.updatePolicyMatrix(q, index, epsilon)
                # S <- S'
                self.state = copy.deepcopy(nextState)
                index = copy.deepcopy(nextIndex)
        return q

    def sumQs(self, q1, q2):
        """Sums 2 Q functions together"""
        q = []
        for i in range(len(q1)):
            row, row1, row2 = [], q1[i], q2[i]
            for k in range(len(row1)):
                list_, list1, list2 = [], row1[k], row2[k]
                for v in range(len(list1)):
                    list_.append(list1[v] + list2[v])
                row.append(list_)
            q.append(row)
        return q

    def doubleqLearning(self, episodes: int = 1, discount: float = 0.9, epsilon: float = 0.1, alpha=0.1):
        """The double Q-learning algorithm
        :returns q1 function & q2 function
        """
        # Initialize Q1
        q1 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # Initialize Q2
        q2 = [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
              [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        # Loop for each episode
        for episode in range(episodes):
            # Initialize S
            self.state = self.doolhof.map[random.randrange(4)][random.randrange(4)]
            index = self.doolhof.coordsToIndex(self.state.position)
            # Loop for each step of episode until S is terminal
            while not self.state.done:
                # Choose A from S using the policy epsilon-greedy in Q1+Q2
                action = random.choices(sorted(list(self.doolhof.action.keys())), self.policy.matrix[index[0]][index[1]])[0]
                # Take action A, observe S'
                nextState = self.doolhof.step(currentState=self.state, action=action)
                nextIndex = self.doolhof.coordsToIndex(nextState.position)
                # Take actionA, observe R
                reward = nextState.reward
                # With 0.5 probability:
                if random.randint(0, 1) > 0:
                    # Q1(S,A) <- Q1(S,A) + alpha(R + gamma * Q2(S', ARGMAXa(Q1(S',a))) - Q1(S,A))
                    a = getMaxFromList(q1[nextIndex[0]][nextIndex[1]])
                    q1[index[0]][index[1]][action] = q1[index[0]][index[1]][action] + alpha * (
                                reward + discount * q2[nextIndex[0]][nextIndex[1]][a] - q1[index[0]][index[1]][action])
                else:
                    # Q2(S,A) <- Q2(S,A) + alpha(R + gamma * Q1(S', ARGMAXa(Q2(S',a))) - Q2(S,A))
                    a = getMaxFromList(q2[nextIndex[0]][nextIndex[1]])
                    q2[index[0]][index[1]][action] = q2[index[0]][index[1]][action] + alpha * (
                                reward + discount * q1[nextIndex[0]][nextIndex[1]][a] - q2[index[0]][index[1]][action])
                # Q = Q1 + Q2
                q = self.sumQs(q1, q2)
                # Update policy using Q1 + Q2
                self.policy.updatePolicyMatrix(q, index, epsilon)
                # S <- S'
                self.state = copy.deepcopy(nextState)
                index = copy.deepcopy(nextIndex)
        return q1, q2

    def __str__(self):
        return "state: %s" % self.state
