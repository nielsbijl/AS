from ModelbasedPredictionAndControl.doolhof.policy import Policy
from ModelbasedPredictionAndControl.doolhof.doolhof import Doolhof
from ModelbasedPredictionAndControl.doolhof.state import State
import copy


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

    def __str__(self):
        return "state: %s" % self.state
