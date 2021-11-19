from ModelbasedPredictionAndControl.doolhof.policy import policy
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
import copy


class agent:
    def __init__(self, doolhof: doolhof, policy: policy, location: tuple):
        self.doolhof = doolhof
        self.policy = policy
        self.location = location

    def coordsToIndex(self, pos: tuple):
        return len(self.doolhof.rewards) - pos[1] - 1, pos[0]

    def indexToCoords(self, index: tuple):
        return index[1], len(self.doolhof.rewards) - index[0] - 1

    def valueFunction(self, location: tuple, policy: policy, discount: int = 1):
        """Een valuefunction dit is een mapping van states naar values.
         Hiervoor kan je dezelfde datastructuur aanhouden als bij de omgeving (e.g. een lijst)."""
        values = []
        # actions = policy.selectAction(location)  # polici iteration
        actions = self.doolhof.action.keys()
        for action in actions:
            pos = self.doolhof.action[action](location)
            index = self.coordsToIndex(pos)
            if (index[0] < 0 or index[1] < 0) or (index[0] > (len(self.doolhof.values)-1) or index[1] > (len(self.doolhof.values)-1)):
                index = self.coordsToIndex(location)
            value = self.doolhof.values[index[0]][index[1]]
            reward = self.doolhof.rewards[index[0]][index[1]]
            values.append(reward + discount * value)
        return max(values)

    def choseAction(self, state, policy):
        """Een functie die een actie kiest op basis van een policy en een state"""
        pass

    def valueIteration(self, threshhold = 0.01):
        """Een implementatie van value iteration"""
        newValues = copy.deepcopy(self.doolhof.values)

        height = len(self.doolhof.values)
        width = len(self.doolhof.values[0])
        for y in range(height):
            for x in range(width):
                value = self.valueFunction(location=(x, y), policy=self.policy, discount=1)
                index = self.coordsToIndex((x, y))
                newValues[index[0]][index[1]] = value
                # oldValue = self.doolhof.values[index[0]][index[1]]
                # delta = max(abs(value) - abs(oldValue), delta)
        return newValues

    def __str__(self):
        return "policy: %s" % self.policy
