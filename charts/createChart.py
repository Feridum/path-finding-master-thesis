import matplotlib.pyplot as plt
import numpy as np

RESULT_PATH = '../results/RL/data/'

def makeChartFromParameters(mazeType, size, strategy, rStrategy, e, t, learningParameters):
    fig, ax = plt.subplots()

    if strategy != 2:
        e = 0.1
        t = 0.1

    for pathName in learningParameters:
            type = str(size) + 'x' + str(size) + '_' + mazeType
            folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])

            file = RESULT_PATH + type + '/' + folderName + '/' + pathName + '.npy'

            [reason, time, epocheNumber, stepsNumber, _, _] = np.load(file, allow_pickle=True)

            ax.plot(stepsNumber[-300:], label=pathName)
            ax.set_ylabel('Number of steps')
            ax.set_xlabel('Epoche number')



    ax.legend(loc='upper right', shadow=True)
    plt.show()

def makeChart(xValues,yValues, xlabel, ylabel, xTicks, labels):
    fig, ax = plt.subplots()

    for i,y in enumerate(yValues):
        ax.plot(xValues, y, label=labels[i])

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    ax.legend(loc='upper right', shadow=True)
    plt.xticks(xValues, xTicks)

    plt.show()

if __name__ == "__main__":
    makeChart(
        [1,2,3],
        [
            [2,3,4],
            [4,6,5],
        ],
        'x', 'y',
        ['January', 'February', 'March'],
        ['1','2']
    )

    strategy = 4
    rStrategy = 'bonus'
    et = 0.2
    size = 100
    mazeType = 'wall'
    learningParameters = ['0.9_0.9']
    # makeChartFromParameters(mazeType=mazeType, size=size, strategy=strategy, rStrategy=rStrategy, e=et, t=et, learningParameters= learningParameters)