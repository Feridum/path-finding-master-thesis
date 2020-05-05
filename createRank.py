import csv
import math

from numpy import load,empty

SUMMARY_PATH = 'results/summary/rank'
DATA_PATH = 'results/data/'

def createRank(size):
    mazes = ['wall', 'columns', 'board']
    variants = ['_change_1', '_change_2', '_change_start']
    files = ['a_star.npy', 'dijkstra.npy', 'bfs.npy']

    data = [x[:] for x in [[None] * (4)] * (3 + 1)]

    column = 1
    data[0][0] = 'maze'

    for f in files:
        row = 1
        data[0][column] = f
        for m in mazes:
                directory = str(s) + 'x' + str(s) + '_' + m
                data[row][0] = directory
                [time, length, _, visited] = load(DATA_PATH + directory + '/' + f, allow_pickle=True)
                dataToWrite = float(time) + float(length)
                sum = 0
                for v in variants:
                    [time, length, _, visited] = load(DATA_PATH + directory + '/' + f, allow_pickle=True)
                    sum = sum + float(time) + float(length)

                data[row][column] = dataToWrite + sum
                row = row + 1
        column = column+1

    with open(SUMMARY_PATH + '_' + str(size) + '.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        rows = zip(data)
        for row in rows:
            filewriter.writerow(row[0])





if __name__ == "__main__":
    for s in [10, 50, 100]:
        createRank(s)