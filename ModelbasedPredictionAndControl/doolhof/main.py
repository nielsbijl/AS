from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy

if __name__ == '__main__':
    rewards = [[-1, -1, -1, 40],
               [-1, -1, -10, -10],
               [-1, -1, -1, -1],
               [10, -2, -1, -1]]
    startPos = (3, 0)
    startReward = 0

    initState = state(startPos, startReward, False)

    doolhof = doolhof(initState, rewards)
    policy = policy(doolhof)

    agent = agent(doolhof, policy, startPos)

    # print(agent.valueFunction(startPos, policy, discount=1))
    print(agent.valueIteration())
    # print(agent.coordsToIndex((2, 0)))
