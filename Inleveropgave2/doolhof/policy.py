import copy

from Inleveropgave2.doolhof.state import State
from Inleveropgave2.doolhof.doolhof import Doolhof
import random


def getMaxFromList(actions: list):
    maxValue = max(actions)
    maxValues = []
    for i in range(len(actions)):
        if actions[i] == maxValue:
            maxValues.append(i)
    return random.choice(maxValues)


class Policy:
    def __init__(self, doolhof: Doolhof):
        self.doolhof = doolhof
        self.matrix = [
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]]]

    def updatePolicyMatrix(self, q: list, index: tuple, epsilon):
        # Get all the possible actions: [0, 1, 2, 3]
        actions = sorted(list(self.doolhof.action.keys()))
        # A* <- argmax Q(St, a)
        aStar = getMaxFromList(q[index[0]][index[1]])
        # For all a ∈ A(St)
        for a in actions:
            if a == aStar:  # if a = A*
                # policyA = 1 - epsilon + epsilon /|A(St)|
                policyA = 1 - epsilon + epsilon / len(q[index[0]][index[1]])
            else:  # if a != A*
                # policyA = epsilon /|A(St)|
                policyA = epsilon / len(q[index[0]][index[1]])
            # π(a|St) <- policyA
            self.matrix[index[0]][index[1]][a] = policyA

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

    def createPolicyMatrix(self, discount: float):
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
