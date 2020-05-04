import csv
import math

from numpy import load,empty

SUMMARY_PATH = 'results/summary/'
DATA_PATH = 'results/data/'

def createSummary(name):
    sizes = [10, 50, 100]
    mazes = ['wall', 'columns', 'board']
    variants = ['', '_change_1', '_change_2', '_change_start']
    files = ['a_star.npy', 'dijkstra.npy', 'bfs.npy']

    data = [x[:] for x in [[None] * (4)] * (3*3*4 + 1)]
    stats = [x[:] for x in [[None] * (4)] * (3 * 3 + 1)]

    sum = 0
    number = 0
    min = +math.inf
    max = -math.inf

    column = 1
    data[0][0] = 'maze'

    for f in files:
        row = 1
        data[0][column] = f
        stats[0][column] = f
        for si, s in enumerate(sizes):
            sum = 0.0
            number = 0.0
            min = +math.inf
            max = -math.inf

            for m in mazes:
                for v in variants:
                    directory = str(s) + 'x' + str(s) + '_' + m + v
                    data[row][0] = directory
                    [time, length, _, visited] = load(DATA_PATH + directory + '/' + f, allow_pickle=True)

                    dataToWrite = time

                    if name == 'length':
                        dataToWrite = length
                    elif name == 'visited':
                        dataToWrite = len(visited)

                    sum = sum + dataToWrite
                    number = number + 1

                    if(min>dataToWrite):
                        min = dataToWrite

                    if(max < dataToWrite):
                        max = dataToWrite

                    data[row][column] = dataToWrite
                    row = row + 1

            stats[si*3 + 1][0] = str(s) + 'avg'
            stats[si*3 + 2][0] = str(s) + 'min'
            stats[si*3 + 3][0] = str(s) + 'max'
            stats[si*3 + 1][column] = sum/number
            stats[si*3 + 2][column] = min
            stats[si*3 + 3][column] = max

        column = column+1

    with open(SUMMARY_PATH + name + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row[0])

    with open(SUMMARY_PATH + name + '_stats.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(stats)
        for row in rows:
            filewriter.writerow(row[0])





if __name__ == "__main__":
    createSummary('time')
    createSummary('length')
    createSummary('visited')