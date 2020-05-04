import sys

from numpy import save, asarray, load
import pathlib
from RL.learnRL import LearnRL
from createResultCsv import createSummaryResultForRLCheck, createSummaryResultForRL
from readMaze import parseMaze

RESULT_PATH = 'results/RL/data/'
RESULT_PATH_CHECK = 'results/RL/data_check/'
IMAGE_PATH = 'results/RL/images/'
RESULT_A_STAR = 'results/data/'


def runSingle(mazeName, strategy, e, t):
    folderName = '_'.join(str(x) for x in [strategy, 'bonus', e, t])
    pathlib.Path(RESULT_PATH + mazeName+'/'+folderName ).mkdir(parents=True, exist_ok=True)
    pathlib.Path(IMAGE_PATH + mazeName+'/'+folderName).mkdir(parents=True, exist_ok=True)

    maze = parseMaze('./maps/txt/' + mazeName + '.txt')

    [_, _, path, _] = load(RESULT_A_STAR + mazeName + '/' + 'a_star.npy', allow_pickle=True)
    learnRL = LearnRL(maze, seed=1584259720496, bonusFields=path)

    for learningRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for futureStepsRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('Start: learningRate:' + str(learningRate) + ' futureStepsRate:' + str(futureStepsRate))
            results = learnRL.startLearn(strategy=strategy, rStrategy=1, learningRate=learningRate,
                                         futureStepsRate=futureStepsRate, e=e, t=t)

            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])
            save(RESULT_PATH + mazeName + '/' + folderName + '/' + pathName + '.npy', asarray(results))

    createSummaryResultForRL(mazeName, strategy=strategy, rStrategy='bonus', e=e, t=t)



def runAll(mazeType, strategy, e, t):
    sizes = [10,50,100]

    for s in sizes:
        fileName = str(s) + 'x' + str(s) + '_' + mazeType
        print('<----- Start' + fileName + '------>')
        runSingle(fileName, strategy=strategy, e=e, t=t)
        print('<----- End' + fileName + '------>')


if __name__ == "__main__":
    print(sys.argv)
    mazeType = sys.argv[1]
    strategy = int(sys.argv[2])
    et = 0.1
    if len(sys.argv) == 4:
        et = float(sys.argv[3])

    if strategy == 1 or strategy == 4:
        runAll(mazeType, strategy=strategy, e=0.1, t=0.1)
    else:
         print('<----- Parameter et' + str(et) + '------>')
         runAll(mazeType, strategy=strategy, e=et, t=et)


