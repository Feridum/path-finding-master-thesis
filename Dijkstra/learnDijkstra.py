import numpy
import math
import time
from collections import namedtuple

Position = namedtuple('Position', 'x y')

class LearnDijkstra:

    def __init__(self, maze):
        self.maze = maze
        self.s = 0
        self.d = []
        self.previous = []
        self.q = []
        self.v = []
        self.f = 0
        self.visited = []

    def get_position(self, position):
        x = position.x
        y = position.y
        return y * self.maze.shape[0] + x

    def resetMaze(self):
        for x in range(self.maze.shape[0]):
            for y in range(self.maze.shape[1]):
                if self.maze[y][x] != '#':
                    self.v.append(self.get_position(Position(x,y)))
                    if self.maze[y][x] == 'E':
                        self.f = self.get_position(Position(x, y))
                    elif self.maze[y][x] == 'B':
                        self.s = self.get_position(Position(x, y))

        self.q = self.v.copy()
        self.d = numpy.full(self.maze.shape[0] * self.maze.shape[1], math.inf, dtype=float)
        self.previous = numpy.full(self.maze.shape[0] * self.maze.shape[1], -math.inf, dtype=float)
        self.d[self.s] = 0
        self.visited = []

    def start(self):
        self.resetMaze()
        start = time.time()
        self.find_path()
        end = time.time()

        previous = self.previous[self.f]
        path = [self.f]
        while previous != -math.inf and previous != self.s:
            path.append(int(previous))
            previous = self.previous[int(previous)]
        return end - start, len(path), path, self.visited

    def find_path(self):
        while len(self.q) > 0:
            u = self.find_min()
            self.visited.append(u)
            neighbours = self.discover_neighbourhood(u)
            if u == self.f:
                break;

            for n in neighbours:
                posN = self.get_position(n)
                if n is not None and self.d[posN] > self.d[u] + 1:
                    self.d[posN] = self.d[u] + 1
                    self.previous[posN] = u



    def find_min(self):
        min = math.inf
        minInd = math.inf
        q = 0
        for index, item in enumerate(self.q):
            if self.d[item] < min:
                min = self.d[item]
                minInd = item
                q = index

        self.q = numpy.delete(self.q, q)
        return minInd


    def discover_neighbourhood(self, indexPosition):
        x = int(indexPosition % self.maze.shape[0])
        y = int(indexPosition // self.maze.shape[0])

        neighbours = set()
        if x - 1 >= 0:
            neighbours.add(self.discover_element(x - 1, y))
        if x + 1 < self.maze.shape[0]:
            neighbours.add(self.discover_element(x + 1, y))
        if y + 1 < self.maze.shape[1]:
            neighbours.add(self.discover_element(x, y+1))
        if y - 1 >= 0:
            neighbours.add(self.discover_element(x, y-1))

        return list(filter(None, list(neighbours)))

    def discover_element(self, x, y):
        if self.maze[y][x] == '#':
            return None
        else:
            return Position(x,y)