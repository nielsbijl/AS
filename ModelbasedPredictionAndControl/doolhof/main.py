from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy

if __name__ == '__main__':
    startPos = (2, 0)
    startReward = 0

    doolhof = doolhof()
    policy = policy(doolhof)

    startStateIndex = doolhof.coordsToIndex(startPos)
    startState = doolhof.map[startStateIndex[0]][startStateIndex[1]]

    agent = agent(doolhof, policy, startState)

    # print(agent.valueCalculate(location=(2, 0), discount=1))
    result = agent.valueIteration(discount=1, threshhold=0.1)
    values = [[i.value for i in row] for row in result]
    # print(values)
    print(doolhof.run(agent, deterministic=True))
    # print(doolhof.values)
    # print(doolhof)


