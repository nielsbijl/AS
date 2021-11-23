from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof


class policy:
    def __init__(self, doolhof: doolhof):
        self.doolhof = doolhof

    def selectAction(self, pos: tuple, discount: float) -> list:
        """select_action die op basis van diens value function en een state een actie terug geeft.
         Voor een mvp kan je beginnen met een random policy. """
        options = {}
        # pos = state.position
        for i in self.doolhof.action.keys():
            index = self.doolhof.coordsToIndex(self.doolhof.action[i](pos))
            if (index[0] < 0 or index[1] < 0) or (index[0] > (len(self.doolhof.values) - 1) or index[1] > (len(self.doolhof.values) - 1)):
                index = self.doolhof.coordsToIndex(pos)
            options[i] = self.doolhof.rewards[index[0]][index[1]] + discount * self.doolhof.values[index[0]][index[1]]

        bestOption = max(options.values())
        finalOptions = []
        for option in options:
            if options[option] == bestOption:
                finalOptions.append(option)
        return finalOptions

    def __str__(self):
        return "doolhof: %s" % self.doolhof
