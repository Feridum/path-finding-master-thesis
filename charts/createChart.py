import matplotlib.pyplot as plt
import numpy as np

RESULT_PATH = '../results/RL/data/'

def makeChartFromParameters(mazeType, size, strategy, rStrategy, e, t):
    fig, ax = plt.subplots()

    for learningRate in [0.9]:
        for futureStepsRate in [0.8]:
            type = str(size) + 'x' + str(size) + '_' + mazeType
            folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])

            file = RESULT_PATH + type + '/' + folderName + '/' + pathName + '.npy'

            [reason, time, epocheNumber, stepsNumber, _, _] = np.load(file, allow_pickle=True)

            ax.plot(stepsNumber, label=pathName)



    ax.legend(loc='upper right', shadow=True)
    plt.show()


if __name__ == "__main__":
    strategy = 2
    rStrategy = 1
    et = 0.1
    size = 100
    mazeType = 'board'
    makeChartFromParameters(mazeType=mazeType, size=size, strategy=strategy, rStrategy=rStrategy, e=et, t=et)