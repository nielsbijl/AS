import unittest
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


class agentTest(unittest.TestCase):
    def testValueIteration(self):
        expected = [[38, 39, 40, 0],
                    [37, 38, 39, 40],
                    [36, 37, 36, 35],
                    [0, 36, 35, 34]]

        startPos = (2, 0)
        startReward = 0

        initState = state(startPos, startReward, False)

        maze = doolhof(initState)
        pol = policy(maze)

        player = agent(maze, pol, startPos)

        result = player.valueIteration(discount=1, threshhold=0.1)

        self.assertEqual(expected, result)

    def testValueCalcuation(self):
        startPos = (2, 0)
        startReward = 0

        initState = state(startPos, startReward, False)

        maze = doolhof(initState)
        pol = policy(maze)

        player = agent(maze, pol, startPos)

        self.assertEqual(-1, player.valueCalculate(location=startPos, discount=1))


if __name__ == '__main__':
    unittest.main()
