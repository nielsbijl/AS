from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof


class policy:
    def __init__(self, doolhof: doolhof):
        self.doolhof = doolhof

    def selectAction(self, pos: tuple, discount: float) -> list:
        """select_action die op basis van diens value function en een state een actie terug geeft.
         Voor een mvp kan je beginnen met een random policy. """
        options = {}
        for i in self.doolhof.action.keys():
            index = self.doolhof.coordsToIndex(self.doolhof.action[i](pos))
            if not self.doolhof.canIGoThere(index):
                index = self.doolhof.coordsToIndex(pos)
            options[i] = self.doolhof.map[index[0]][index[1]].reward + discount * self.doolhof.map[index[0]][index[1]].value
        bestOption = max(options.values())
        finalOptions = []
        for option in options:
            if options[option] == bestOption:
                finalOptions.append(option)
        return finalOptions

    def __str__(self):
        return "doolhof: %s" % self.doolhof
