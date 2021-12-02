from Inleveropgave2.doolhof.doolhof import Doolhof
from Inleveropgave2.doolhof.agent import Agent
from Inleveropgave2.doolhof.policy import Policy


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

    doolhof = Doolhof()
    policy = Policy(doolhof)

    startStateIndex = doolhof.coordsToIndex(startPos)
    startState = doolhof.map[startStateIndex[0]][startStateIndex[1]]

    agent = Agent(doolhof, policy, startState)

    # doolhof.run(agent, deterministic=True, discount=1)

    print("policy: ", agent.policy.selectAllActions(discount=1))
    #[[[1], [1], [1], [None]], [[0, 1], [0], [0], [0]], [[0, 1], [0], [3], [3]], [[None], [0], [0], [0, 3]]]

    print(agent.monteCarloPolicyEvaluation(episodes=50000, discount=1))

