import unittest
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


class agentTest(unittest.TestCase):
    def testRun(self):
        expected = [(2, 0), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3)]

        startPos, startReward = (2, 0), 0
        initState = state(startPos, startReward, False)

        maze = doolhof(initState)
        pol = policy(maze)
        player = agent(maze, pol, startPos)

        result = maze.run(agent=player, deterministic=True)

        self.assertEqual(expected, result)

    def testCoordsToIndex(self):
        startPos, startReward = (2, 0), 0
        initState = state(startPos, startReward, False)

        maze = doolhof(initState)

        self.assertEqual((3, 0), maze.coordsToIndex((0, 0)))


    def testIndexToCoords(self):
        startPos, startReward = (2, 0), 0
        initState = state(startPos, startReward, False)

        maze = doolhof(initState)

        self.assertEqual((0, 0), maze.indexToCoords((3, 0)))


if __name__ == '__main__':
    unittest.main()
