from ModelbasedPredictionAndControl.doolhof.doolhof import doolhof
from ModelbasedPredictionAndControl.doolhof.agent import agent
from ModelbasedPredictionAndControl.doolhof.policy import policy


def printPolicy(policyMatrix: list):
    string = [str(row) for row in policyMatrix]
    for i in range(len(string)):
        string[i] = string[i].replace('0', '↑')
        string[i] = string[i].replace('1', '→')
        string[i] = string[i].replace('2', '↓')
        string[i] = string[i].replace('3', '←')

    string = list(string)
    [print(stringRow) for stringRow in string]


if __name__ == '__main__':
    startPos = (2, 0)

    doolhof = doolhof()
    policy = policy(doolhof)

    startStateIndex = doolhof.coordsToIndex(startPos)
    startState = doolhof.map[startStateIndex[0]][startStateIndex[1]]

    agent = agent(doolhof, policy, startState)

    print("Route:", doolhof.run(agent, deterministic=True, discount=1))
    result = doolhof.map
    values = [[i.value for i in row] for row in result]
    print("\nValue funtion:")
    [print(valueRow) for valueRow in values]

    print("\n Policy:")
    printPolicy(agent.policy.selectAllActions(discount=1))
