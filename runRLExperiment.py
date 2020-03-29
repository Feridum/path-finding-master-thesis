from numpy import save, asarray, load
import pathlib
from RL.learnRL import LearnRL
from createResultCsv import createSummaryResultForRLCheck, createSummaryResultForRL
from readMaze import parseMaze

RESULT_PATH = 'results/RL/data/'
RESULT_PATH_CHECK = 'results/RL/data_check/'
IMAGE_PATH = 'results/RL/images/'


def runSingle(mazeName, strategy=1, rStrategy=1, e=0.1, t=0.5):
    pathlib.Path(RESULT_PATH + mazeName).mkdir(parents=True, exist_ok=True)
    pathlib.Path(IMAGE_PATH + mazeName).mkdir(parents=True, exist_ok=True)

    maze = parseMaze('./maps/txt/' + mazeName + '.txt')

    learnRL = LearnRL(maze, seed=1584259720496)

    for learningRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for futureStepsRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('Start: learningRate:' + str(learningRate) + ' futureStepsRate:' + str(futureStepsRate))
            results = learnRL.startLearn(strategy=strategy, rStrategy=rStrategy, learningRate=learningRate,
                                         futureStepsRate=futureStepsRate, e=e, t=t)
            print(results[0])
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate, strategy, rStrategy, e, t])
            save(RESULT_PATH + mazeName + '/' + pathName + '.npy', asarray(results))

    createSummaryResultForRL(mazeName, strategy=strategy, rStrategy=rStrategy, e=e, t=t)


def runCheck(mazeName, newMaze, strategy=1, rStrategy=1, e=0.1, t=0.5):
    pathlib.Path(RESULT_PATH_CHECK + mazeName).mkdir(parents=True, exist_ok=True)

    maze = parseMaze('./maps/txt/' + newMaze + '.txt')

    learnRL = LearnRL(maze, seed=1584259720496)

    for learningRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for futureStepsRate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('Start: learningRate:' + str(learningRate) + ' futureStepsRate:' + str(futureStepsRate))
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate, strategy, rStrategy, e, t])
            [_, _, _, _, Q, R] = load(RESULT_PATH + mazeName + '/' + pathName + '.npy', allow_pickle=True)
            results = learnRL.startFindPath(Q, R, strategy=strategy, rStrategy=rStrategy, learningRate=learningRate,
                                         futureStepsRate=futureStepsRate, e=e, t=t)
            save(RESULT_PATH_CHECK + newMaze + '/' + pathName + '.npy', asarray(results))

    createSummaryResultForRLCheck(newMaze, strategy=strategy, rStrategy=rStrategy, e=e, t=t)

def runAll(mazeType):
    # sizes = [10,50,100]
    sizes = [10]

    for s in sizes:
        runSingle(str(s) + 'x' + str(s) + '_' + mazeType)

def runAllCheck(mazeType, newMaze):
    # sizes = [10,50,100]
    sizes = [10]

    for s in sizes:
        runCheck(str(s) + 'x' + str(s) + '_' + mazeType, str(s) + 'x' + str(s) + '_' + newMaze)


if __name__ == "__main__":
    # runAll('wall')
    runAllCheck('wall', 'wall')
