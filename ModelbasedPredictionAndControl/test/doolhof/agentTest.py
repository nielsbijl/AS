import unittest
from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


class agentTest(unittest.TestCase):
    def testValueOneIteration(self):
        expected = [[-1, -1, 7, 17.25],
                    [-1, -3.25, -3.25, 4.75],
                    [1.75, -1.25, -3.25, -3.25],
                    [4.25, 1.5, -1.25, -1]]

        rewards = [[-1, -1, -1, 40],
                   [-1, -1, -10, -10],
                   [-1, -1, -1, -1],
                   [10, -2, -1, -1]]
        startPos = (2, 3)
        startReward = 0

        initState = state(startPos, startReward, False)

        maze = doolhof(initState, rewards)
        pol = policy(maze)

        agnt = agent(maze, pol, startPos)

        # print(agent.valueFunction(startPos, policy, discount=1))
        result = agnt.valueIteration()

        self.assertEqual(expected, result)


if __name__ == '__main__':
    unittest.main()
