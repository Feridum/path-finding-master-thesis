import time

import numpy
from collections import namedtuple
from queue import PriorityQueue

Position = namedtuple('Position', 'x y')


class AStarNode:
    def __init__(self, position, f, g, parent=None):
        self.parent = parent
        self.position = position
        self.f = f
        self.g = g

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f

class LearnAStar:

    def __init__(self, maze):
        self.maze = maze
        start = numpy.where(maze == 'B')
        end = numpy.where(maze == 'E')
        self.startPosition = Position(start[1][0], start[0][0])
        self.endPosition = Position(end[1][0], end[0][0])

    def resetMaze(self):
        self.openList = PriorityQueue()
        self.closeList = set()
        self.openList.put((0, AStarNode(self.startPosition, 0, 0)))
        self.target = None

    def start(self):
        self.resetMaze()
        start = time.time()
        self.find_path()
        end = time.time()

        path = []
        i = self.target
        while i.parent is not None:
            path.append(self.get_position(i.position))
            i = i.parent

        return end - start, len(path), path

    def find_path(self):
        while True:
            node = self.openList.get()[1]
            position = self.get_position(node.position)

            if node.position == self.endPosition:
                self.target = node
                break

            if position not in self.closeList:
                self.closeList.add(position)
                neighbours = self.discover_neighbourhood(node.position)

                for n in neighbours:
                    if self.get_position(n) not in self.closeList:
                        f = self.calculate_f(n, node.g)
                        newNode = AStarNode(n, f, node.g + 1, node)
                        self.openList.put((f, newNode))

    def calculate_f(self, position, current_g):
        return abs(position.x - self.endPosition.x) + abs(position.y - self.endPosition.y) + current_g + 1

    def discover_neighbourhood(self, position):
        x = position.x
        y = position.y

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
        elif self.maze[y][x] == 'E':
            return Position(x,y)
        else:
            return Position(x,y)

    def get_position(self, position):
        x = position.x
        y = position.y
        return y * self.maze.shape[0] + x
