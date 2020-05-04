import csv
import math
import pathlib

from numpy import load,empty, average

SUMMARY_PATH = 'results/RL/summary/check/'
DATA_PATH = 'results/RL/data_check/'

def createSummary(name, st , size, v):
    mazes = ['wall', 'columns', 'board']
    strategies = [
        '1_{}_0.1_0.1', '2_{}_0.2_0.2', '2_{}_0.4_0.4', '2_{}_0.6_0.6', '4_{}_0.1_0.1'
    ]
    data = [x[:] for x in [[None] * (3*len(strategies)+1)] * (9 * 9 + 1)]
    pathlib.Path(SUMMARY_PATH + st).mkdir(parents=True, exist_ok=True)
    column = 1
    data[0][0] = 'maze'



    for m in mazes:
        for s in strategies:
            row = 1
            directory = str(size) + 'x' + str(size) + '_' + m + v
            data[0][column] = m + '_' + s.format(st)
            for lIndex, learningRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                for fIndex, futureStepsRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                        data[row][0] = str(learningRate) + '_' + str(futureStepsRate)
                        folderName = s.format(st)
                        pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])
                        [time, stepsNumber, reason] = load(DATA_PATH + directory + '/' + folderName + '/' + pathName + '.npy', allow_pickle=True)

                        dataToWrite = time

                        if name == 'stepsNumber':
                            dataToWrite = stepsNumber
                        elif name == 'reason':
                            dataToWrite = reason

                        data[row][column] = dataToWrite
                        row = row + 1

            column = column+1

    with open(SUMMARY_PATH + st + '/' + str(size) + '_' + name + v +'.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row[0])





if __name__ == "__main__":
    versions = ['_change_1', '_change_2', '_change_start']
    strategies = ['1', '2', 'bonus']
    for s in [10,50,100]:
        for st in strategies:
            for v in versions:
                createSummary('time', st, s,v)
                createSummary('stepsNumber', st, s,v)