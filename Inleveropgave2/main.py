from Inleveropgave2.doolhof.maze import Maze
from Inleveropgave2.doolhof.agent import Agent
from Inleveropgave2.doolhof.policy import Policy
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np


def printPolicy(policyMatrix: list):
    string = [str(row) for row in policyMatrix]
    for i in range(len(string)):
        string[i] = string[i].replace('0', '↑')
        string[i] = string[i].replace('1', '→')
        string[i] = string[i].replace('2', '↓')
        string[i] = string[i].replace('3', '←')

    string = list(string)
    [print(stringRow) for stringRow in string]


def triangulation_for_triheatmap(M, N):
    """BRON: https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4
    -triangles """
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]


def reformatMatrix(values):
    """Reformat the matrix so the matrix can be ploted with triangles"""
    zuid, oost, noord, west = [], [], [], []
    for i in range(-1, len(values) * -1 - 1, -1):
        zuidRow, oostRow, noordRow, westRow = [], [], [], []
        row = values[i]
        for square in row:
            zuidRow.append(square[2])
            oostRow.append(square[1])
            noordRow.append(square[0])
            westRow.append(square[3])
        zuid.append(zuidRow)
        oost.append(oostRow)
        noord.append(noordRow)
        west.append(westRow)
    return [zuid, oost, noord, west]


def printTriangleMatrix(values):
    """BRON: https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4
    -triangles """
    triangul = triangulation_for_triheatmap(4, 4)
    fig, ax = plt.subplots()
    imgs = [ax.tripcolor(t, np.ravel(val), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
            for t, val in zip(triangul, values)]
    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(4):
            for j in range(4):
                v = val[j][i]
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w',
                        ha='center', va='center')
    plt.show()


if __name__ == '__main__':
    startPos = (2, 0)

    doolhof = Maze()
    policy = Policy(doolhof)

    startStateIndex = doolhof.coordsToIndex(startPos)
    startState = doolhof.map[startStateIndex[0]][startStateIndex[1]]

    agent = Agent(doolhof, policy, startState)

    values = agent.doubleqLearning(episodes=100000, discount=1)[0]
    matrix = reformatMatrix(values)
    printTriangleMatrix(matrix)

    values = agent.policy.matrix
    matrix = reformatMatrix(values)
    printTriangleMatrix(matrix)
