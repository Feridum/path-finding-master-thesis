import numpy
import math
import time
from collections import namedtuple

Position = namedtuple('Position', 'x y')

class LearnBFS:

    def __init__(self, maze):
        start = numpy.where(maze == 'B')
        end = numpy.where(maze == 'E')
        self.maze = maze
        self.s = self.get_position(Position(start[1], start[0]))[0]
        self.f = self.get_position(Position(end[1], end[0]))[0]
        self.d = []
        self.previous = []
        self.q = []
        self.v = []
        self.visited = []

    def get_position(self, position):
        x = position.x
        y = position.y
        return y * self.maze.shape[0] + x

    def resetMaze(self):
        self.q = []
        self.parent = numpy.full(self.maze.shape[0] * self.maze.shape[1], -math.inf, dtype=float)
        self.visited = numpy.full(self.maze.shape[0] * self.maze.shape[1], False, dtype=bool)
        self.q.append(self.s)
        self.visited[self.s] = True

    def start(self):
        self.resetMaze()
        start = time.time()
        self.find_path()
        end = time.time()

        parent = self.parent[self.f]
        path = [self.f]
        while parent != -math.inf and parent != self.s:
            path.append(int(parent))
            parent = self.parent[int(parent)]

        visited = []
        for index, item in enumerate(self.visited):
            if item:
                visited.append(index)

        return end - start, len(path), path, visited

    def find_path(self):
        while len(self.q) > 0:
            u = self.q.pop(0)

            if u == self.f:
                break;

            neighbours = self.discover_neighbourhood(u)

            for n in neighbours:
                posN = self.get_position(n)
                if not self.visited[posN]:
                    self.visited[posN] = True
                    self.parent[posN] = u
                    self.q.append(posN)


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