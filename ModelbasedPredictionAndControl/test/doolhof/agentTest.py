import unittest
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


class agentTest(unittest.TestCase):
    def setUp(self):
        self.startPos = (2, 0)

        self.maze = doolhof()
        self.pol = policy(self.maze)

        startStateIndex = self.maze.coordsToIndex(self.startPos)
        startState = self.maze.map[startStateIndex[0]][startStateIndex[1]]

        self.player = agent(self.maze, self.pol, startState)

    def testValueIteration(self):
        expected = [[38, 39, 40, 0],
                    [37, 38, 39, 40],
                    [36, 37, 36, 35],
                    [0, 36, 35, 34]]

        result = self.player.valueIteration(discount=1, threshhold=0.1)
        values = [[i.value for i in row] for row in result]

        self.assertEqual(expected, values)

    def testValueCalcuation(self):
        self.assertEqual(-1, self.player.valueCalculate(location=self.startPos, discount=1))


if __name__ == '__main__':
    unittest.main()
