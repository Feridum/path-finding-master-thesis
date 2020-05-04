import sys

from numpy import save, asarray, load
import pathlib
from RL.learnRL import LearnRL
from createResultCsv import createSummaryResultForRLCheck, createSummaryResultForRL
from readMaze import parseMaze

RESULT_PATH = 'results/RL/data/'
RESULT_PATH_CHECK = 'results/RL/data_check/'
IMAGE_PATH = 'results/RL/images/'

def runCheck(mazeName, newMaze, strategy, rStrategy, e, t):
    folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])
    pathlib.Path(RESULT_PATH_CHECK + newMaze +'/'+folderName).mkdir(parents=True, exist_ok=True)

    maze = parseMaze('./maps/txt/' + newMaze + '.txt')

    learnRL = LearnRL(maze, seed=1584259720496, maxTime=5)

    for learningRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for futureStepsRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('Start: learningRate:' + str(learningRate) + ' futureStepsRate:' + str(futureStepsRate))
            folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])

            [_, _, _, _, Q, R] = load(RESULT_PATH + mazeName + '/' + folderName + '/' + pathName + '.npy', allow_pickle=True)

            results = learnRL.startFindPath(Q, R, strategy=1, rStrategy=1, learningRate=0.5,
                                         futureStepsRate=0.5, e=0.1, t=0.1)
            save(RESULT_PATH_CHECK + newMaze + '/' + folderName + '/' + pathName + '.npy', asarray(results))

    createSummaryResultForRLCheck(newMaze, strategy=strategy, rStrategy=rStrategy, e=e, t=t)

def runAllCheck(mazeType, strategy, rStrategy, e, t):
    sizes = [10,50,100]

    variants = ['_change_1', '_change_2', '_change_start']

    for s in sizes:
        for v in variants:
            fileName = str(s) + 'x' + str(s) + '_' + mazeType
            checkFileName = str(s) + 'x' + str(s) + '_' + mazeType+v
            print('<----- Start' + checkFileName + '------>')
            runCheck(fileName, checkFileName, strategy=strategy, rStrategy=rStrategy, e=e, t=t)
            print('<----- End' + checkFileName + '------>')


if __name__ == "__main__":
    mazeType = ['wall', 'columns', 'board']
    strategies = [1,2,4]
    rStrategy = sys.argv[1]

    for s in strategies:
        print('<----- Start strategy' + str(s) + '------>')
        for m in mazeType:
            if s == 1 or s == 4:
                runAllCheck(m, strategy=s, rStrategy=rStrategy, e=0.1, t=0.1)
            else:
                for et in [0.2, 0.4, 0.6]:
                    print('<----- Parameter et' + str(et) + '------>')
                    runAllCheck(m, strategy=s, rStrategy=rStrategy, e=et, t=et)


