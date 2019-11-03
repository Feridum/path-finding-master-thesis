import numpy


def parseMaze(filename):
    f = open(filename)
    [row, col] = f.readline().split(" ")
    maze = numpy.zeros([int(row), int(col)], dtype=str)

    row = 0
    col = 0
    for line in f.readlines():
        for c in line:
            if c.isprintable():
                maze[row][col] = c
                col = col +1
        row = row + 1
        col = 0

    print(maze)
    return maze


