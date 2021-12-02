import copy

from Inleveropgave2.doolhof.state import State
from Inleveropgave2.doolhof.doolhof import Doolhof


class Policy:
    def __init__(self, doolhof: Doolhof):
        self.doolhof = doolhof
        self.matrix = [[[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                       [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                       [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]],
                       [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]]

    def selectAction(self, pos: tuple, discount: float) -> list:
        """This function calculate the best action to execute in its current position on the basis of the
        valuefunction, it's possible to return multiply best actions"""
        options = {}
        for i in self.doolhof.action.keys():
            index = self.doolhof.coordsToIndex(self.doolhof.action[i](pos))
            if not self.doolhof.canIGoThere(index):
                index = self.doolhof.coordsToIndex(pos)
            options[i] = self.doolhof.map[index[0]][index[1]].reward + discount * self.doolhof.map[index[0]][
                index[1]].value
        bestOption = max(options.values())
        finalOptions = []
        for option in options:
            if options[option] == bestOption:
                finalOptions.append(option)
        return finalOptions

    def selectAllActions(self, discount: float):
        """This function calculates best actions for all the states (policy)"""
        for y in range(len(self.doolhof.map)):
            for x in range(len(self.doolhof.map[0])):
                pol = self.selectAction((x, y), discount)
                index = self.doolhof.coordsToIndex((x, y))
                if self.doolhof.map[index[0]][index[1]].done:
                    self.matrix[index[0]][index[1]] = [None]
                else:
                    self.matrix[index[0]][index[1]] = pol
        return self.matrix

    def __str__(self):
        return "doolhof: %s" % self.doolhof
