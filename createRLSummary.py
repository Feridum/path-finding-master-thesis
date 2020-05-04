import csv
import math

from numpy import load,empty, average

SUMMARY_PATH = 'results/RL/summary/'
DATA_PATH = 'results/RL/data/'

def createSummary(name, versions , s):
    mazes = ['wall', 'columns', 'board']

    data = [x[:] for x in [[None] * (3*len(versions)+1)] * (9 * 9 + 1)]

    column = 1
    data[0][0] = 'maze'

    for m in mazes:
        for v in versions:
            row = 1
            directory = str(s) + 'x' + str(s) + '_' + m
            data[0][column] = m + '_' +v
            for lIndex, learningRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                for fIndex, futureStepsRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
                        data[row][0] = str(learningRate) + '_' + str(futureStepsRate)

                        pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])
                        [reason, time, epocheNumber, stepsNumber, _, _] = load(DATA_PATH + directory + '/' + v + '/' + pathName + '.npy', allow_pickle=True)

                        dataToWrite = time

                        if name == 'stepsNumber':
                            dataToWrite = stepsNumber[-1]
                        elif name == 'epocheNumber':
                            dataToWrite = epocheNumber
                        elif name == 'reason':
                            dataToWrite = reason
                        elif name == 'avg':
                            dataToWrite = average(stepsNumber)

                        data[row][column] = dataToWrite
                        row = row + 1

            column = column+1

    with open(SUMMARY_PATH + str(s) + '_' + name + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row[0])





if __name__ == "__main__":
    versions = ['1_bonus_0.1_0.1', '2_bonus_0.2_0.2', '2_bonus_0.4_0.4', '2_bonus_0.6_0.6', '4_bonus_0.1_0.1']
    for s in [10,50,100]:
        createSummary('avg', versions, s)
        createSummary('time', versions, s)
        createSummary('stepsNumber', versions, s)
        createSummary('reason', versions, s)
        createSummary('epocheNumber', versions, s)