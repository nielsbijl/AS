from ModelbasedPredictionAndControl.doolhof.policy import policy
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
import copy


class agent:
    def __init__(self, doolhof: doolhof, policy: policy, location: tuple):
        self.doolhof = doolhof
        self.policy = policy
        self.location = location

    def bellmanEquation(self, location: tuple, action: int, discount: float):
        pos = self.doolhof.action[action](location)
        index = self.doolhof.coordsToIndex(pos)
        if not self.doolhof.canIGoThere(index):
            index = self.doolhof.coordsToIndex(location)
        value = self.doolhof.values[index[0]][index[1]]
        reward = self.doolhof.rewards[index[0]][index[1]]
        return reward + discount * value

    def valueCalculate(self, location: tuple, iteration: int, discount: float = 1, deterministic: bool = False):
        """Een valuefunction dit is een mapping van states naar values.
         Hiervoor kan je dezelfde datastructuur aanhouden als bij de omgeving (e.g. een lijst)."""
        values = []
        actions = self.doolhof.action.keys()
        for action in actions:
            value = self.bellmanEquation(location, action, discount)
            # if not deterministic:
            #     sum = 0
            #     for secondAction in actions:
            #         pos = self.doolhof.action[secondAction](location)
            #         index = self.doolhof.coordsToIndex(pos)
            #         if not self.doolhof.canIGoThere(index):
            #             index = self.doolhof.coordsToIndex(location)
            #         value = self.doolhof.values[index[0]][index[1]]
            #         reward = self.doolhof.rewards[index[0]][index[1]]
            #         sum += self.doolhof.actionChance[secondAction] * (reward + discount * value)
            #     values.append(sum)
            # else:
            #     values.append(reward + discount * value)
            values.append(value)

        return max(values)

    def choseAction(self, state, policy):
        """Een functie die een actie kiest op basis van een policy en een state"""
        pass

    def valueIteration(self, discount: float, threshhold=0.01):
        """Een implementatie van value iteration"""
        newValues = copy.deepcopy(self.doolhof.values)
        done = False
        iteration = 0
        while not done:
            delta = 0
            height = len(self.doolhof.values)
            width = len(self.doolhof.values[0])
            for y in range(height):
                for x in range(width):
                    index = self.doolhof.coordsToIndex((x, y))
                    value = self.valueCalculate(location=(x, y), discount=discount,
                                                iteration=iteration)
                    if self.doolhof.end[index[0]][index[1]]:
                        value = 0
                    newValues[index[0]][index[1]] = value
                    oldValue = self.doolhof.values[index[0]][index[1]]
                    # delta = max(abs(value) - abs(oldValue), delta)
                    delta = max(delta, abs(oldValue - value))
            self.doolhof.values = copy.deepcopy(newValues)
            if delta < threshhold:
                done = True
            iteration += 1
        return newValues

    def __str__(self):
        return "policy: %s" % self.policy
