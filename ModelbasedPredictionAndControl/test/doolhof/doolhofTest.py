import unittest
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


class agentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.startPos = (2, 0)

        self.maze = doolhof()
        self.pol = policy(self.maze)

        startStateIndex = self.maze.coordsToIndex(self.startPos)
        self.startState = self.maze.map[startStateIndex[0]][startStateIndex[1]]

        self.player = agent(self.maze, self.pol, self.startState)

    def testRun(self):
        expected = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]
        result = self.maze.run(agent=self.player, deterministic=True)
        self.assertEqual(expected, result)

    def testCoordsToIndex(self):
        self.assertEqual((3, 0), self.maze.coordsToIndex((0, 0)))

    def testIndexToCoords(self):
        self.assertEqual((0, 0), self.maze.indexToCoords((3, 0)))


if __name__ == '__main__':
    unittest.main()
