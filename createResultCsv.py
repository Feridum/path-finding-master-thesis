import os
import pathlib

from numpy import load,empty, asarray
import csv

RESULT_PATH = 'results/data/'
CSV_PATH = 'results/csv/'

RL_RESULT_PATH = 'results/RL/data/'
RL_CSV_PATH = 'results/RL/csv/'

RL_CHECK_RESULT_PATH = 'results/RL/data_check/'
RL_CHECK_CSV_PATH = 'results/RL/csv_check/'


def createSummaryResult(directory, files):

    data = empty([3,3])

    for index, f in enumerate(files):
        [time, length, _, visited] = load(RESULT_PATH+directory+'/'+f+'.npy', allow_pickle=True)
        data[0][index] = length
        data[1][index] = time
        data[2][index] = len(visited)


    with open(CSV_PATH + directory + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        filewriter.writerow(['Element', files[0], files[1], files[2]])
        filewriter.writerow(['Długość', data[0][0], data[0][1], data[0][2]])
        filewriter.writerow(['Czas', data[1][0], data[1][1], data[1][2]])
        filewriter.writerow(['Odwiedzone', data[2][0], data[2][1], data[2][2]])


def createSummaryResultForRL(directory, strategy, rStrategy, e, t):
    folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])
    numberOfResults = len([name for name in os.listdir(RL_RESULT_PATH + directory+'/'+folderName)])
    pathlib.Path(RL_CSV_PATH + folderName).mkdir(parents=True, exist_ok=True)

    data = [x[:] for x in [[None] * (numberOfResults+1)] * 6]

    data[0][0] = 'learningRate'
    data[1][0] = 'futureStepsRate'
    data[2][0] = 'Status'
    data[3][0] = 'Czas'
    data[4][0] = 'Liczba epok'
    data[5][0] = 'Dlugosc sciezki'

    for lIndex, learningRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        for fIndex, futureStepsRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])
            [reason, time, epocheNumber, stepsNumber, _, _] = load(RL_RESULT_PATH + directory + '/' + folderName + '/' + pathName + '.npy', allow_pickle=True)
            index = lIndex * 9 + fIndex
            data[0][index+1] = learningRate
            data[1][index+1] = futureStepsRate
            data[2][index+1] = reason
            data[3][index+1] = time
            data[4][index+1] = epocheNumber
            data[5][index+1] = stepsNumber[-1]



    with open(RL_CSV_PATH + folderName + '/'+directory + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row)

def createSummaryResultForRLCheck(directory, strategy, rStrategy, e, t):
    folderName = '_'.join(str(x) for x in [strategy, rStrategy, e, t])
    numberOfResults = len([name for name in os.listdir(RL_CHECK_RESULT_PATH + directory+'/'+folderName)])
    pathlib.Path(RL_CHECK_CSV_PATH + folderName).mkdir(parents=True, exist_ok=True)
    data = [x[:] for x in [[None] * (numberOfResults+1)] * 5]

    data[0][0] = 'learningRate'
    data[1][0] = 'futureStepsRate'
    data[2][0] = 'Czas'
    data[3][0] = 'Dlugosc sciezki'
    data[4][0] = 'Powod'

    for lIndex, learningRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
        for fIndex, futureStepsRate in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            pathName = '_'.join(str(x) for x in [learningRate, futureStepsRate])
            [time, stepsNumber, reason] = load(RL_CHECK_RESULT_PATH + directory + '/' + folderName + '/' + pathName + '.npy', allow_pickle=True)
            index = lIndex * 9 + fIndex
            data[0][index+1] = learningRate
            data[1][index+1] = futureStepsRate
            data[2][index+1] = time
            data[3][index+1] = stepsNumber
            data[4][index+1] = reason



    with open(RL_CHECK_CSV_PATH + folderName + '/' + directory + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row)

if __name__ == "__main__":
    createSummaryResultForRL('10x10_wall', strategy= 1, rStrategy= 1, e= 0.1,t= 0.5)