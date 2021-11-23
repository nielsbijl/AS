from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.state import state
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy

if __name__ == '__main__':
    startPos = (2, 0)
    startReward = 0

    initState = state(startPos, startReward, False)

    doolhof = doolhof(initState)
    policy = policy(doolhof)

    agent = agent(doolhof, policy, startPos)

    # print(agent.valueFunction(startPos, policy, discount=1))
    # print(agent.valueIteration(discount=0.9, threshhold=0.1))
    doolhof.run(agent, deterministic=True)
    # print(doolhof.values)
