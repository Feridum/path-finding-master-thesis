import math
import random
import time

import numpy
from collections import namedtuple
from enum import Enum

from displayPath import visualizeRL

Position = namedtuple('Position', 'x y')


class Direction(Enum):
    TOP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class LearnRL:
    def __init__(self, maze, bonusFields = []):
        self.start = numpy.where(maze == 'B')
        end = numpy.where(maze == 'E')
        self.startPosition = Position(self.start[1], self.start[0])
        self.maze = maze
        self.endPosition = Position(end[1], end[0])
        self.learningRate = 0.8
        self.futureStepsRate = 0.9
        self.strategy = 1
        self.R = []
        self.Q = []
        self.P = []
        self.currentPosition = Position(0,0)
        self.stepsNumber = []
        self.maxDistance = 0
        self.bonusFields = bonusFields

    def resetMaze(self):
        shape = self.maze.shape
        self.maxDistance = shape[0] + shape[1]
        self.R = numpy.zeros([shape[0] * shape[1], 4])
        self.P = numpy.zeros([shape[0] * shape[1], 4])
        self.Q = numpy.full([shape[0] * shape[1], 4], -math.inf, dtype=float)
        # self.Q[self.get_position(self.endPosition)] = numpy.full(4, 0)
        self.currentPosition = Position(self.start[1], self.start[0])
        self.stepsNumber = []

    def startLearn(self, strategy = 1):
        self.resetMaze()
        self.strategy = strategy
        start = time.time()
        self.learn()
        end = time.time()
        self.saveQRToFile()
        return end-start, len(self.stepsNumber), self.stepsNumber
        # return end-start, self.stepsNumber, self.Q, self.R

    def learn(self):
        shape = self.maze.shape
        self.P = numpy.zeros([shape[0] * shape[1], 4])
        steps_number = 0
        while self.currentPosition != self.endPosition:
            self.discover_neighbourhood()
            self.make_next_move(self.get_next_move())
            steps_number = steps_number + 1

        self.stepsNumber = numpy.append(self.stepsNumber, steps_number)

        print(steps_number)
        if self.stepsNumber.shape[0] > 3:
            if self.stepsNumber[-1] == self.stepsNumber[-2] and self.stepsNumber[-2] == self.stepsNumber[-3]:
                print('finished')
                 # print(self.Q)
            elif len(self.stepsNumber) > 500:
                print('max epoches reached')
            else:
                self.currentPosition = self.startPosition
                self.learn()
        else:
            self.currentPosition = self.startPosition
            self.learn()

    def startFindPath(self, Q, R, strategy=1):
        self.resetMaze()
        self.strategy = strategy
        self.Q = Q
        self.R = R
        start = time.time()
        stepsNumber = self.findPath()
        end = time.time()
        self.saveQRToFile()
        return end - start, stepsNumber

    def saveQRToFile(self):
        numpy.save('./R.npy', self.R)
        numpy.save('./Q.npy', self.Q)

    def findPath(self):
        steps_number = 0
        while self.currentPosition != self.endPosition:
            self.discover_neighbourhood()
            self.make_next_move(self.get_next_move())
            steps_number = steps_number + 1
        return steps_number

    def discover_neighbourhood(self):
        x = self.currentPosition.x[0]
        y = self.currentPosition.y[0]
        position = self.get_position(self.currentPosition)

        self.R[position] = numpy.full(4, -math.inf)

        if x - 1 >= 0:
            self.discover_element(x - 1, y, Direction.LEFT)
        if x + 1 < self.maze.shape[0]:
            self.discover_element(x + 1, y, Direction.RIGHT)
        if y + 1 < self.maze.shape[1]:
            self.discover_element(x, y + 1, Direction.DOWN)
        if y - 1 >= 0:
            self.discover_element(x, y - 1, Direction.TOP)

    def discover_element(self, x, y, direction):
        position = self.get_position(self.currentPosition)
        if self.maze[y][x] == '#':
            self.R[position][direction.value] = -math.inf
        elif self.maze[y][x] == 'E':
            self.R[position][direction.value] = 100
        elif position in self.bonusFields:
            self.R[position][direction.value] = -0.5
        else:
            # if self.strategy == 1:
                self.R[position][direction.value] = -1
                # self.R[position][direction.value] = -5 + (1/self.calculate_distance_to_end2(self.get_next_position(direction)))
            # elif self.strategy == 2:
            #     self.R[position][direction.value] = (1/self.calculate_distance_to_end(self.get_next_position(direction)))

    def get_q_row(self, position):
        q = self.Q[position]
        for i in range(4):
            if self.R[position][i] == -math.inf:
                q[i] = -math.inf
            elif self.R[position][i] == 100 and self.Q[position][i] == 0:
                q[i] = 100
            elif self.Q[position][i] == -math.inf and self.R[position][i] != -math.inf:
                q[i] = self.R[position][i]
        return q

    def get_position(self, position):
        x = position.x[0]
        y = position.y[0]
        return y * self.maze.shape[0] + x

    def greedy_strategy(self):
        position = self.get_position(self.currentPosition)
        max_indices = numpy.argwhere(self.get_q_row(position) == self.get_max_value(position))
        return Direction(random.choice(max_indices))

    def calculate_distance_to_end(self, position):
        return abs(position.x - self.endPosition.x) + abs(position.y - self.endPosition.y)

    def calculate_distance_to_end2(self, position):
        return math.sqrt(pow((position.x - self.endPosition.x),2) + pow((position.y - self.endPosition.y),2))

    def a_star_like(self):
        position = self.get_position(self.currentPosition)
        max_indices = numpy.argwhere(self.get_q_row(position) == self.get_max_value(position))
        min = []
        minDist = math.inf
        for i in max_indices:
            newPosition = self.get_next_position(Direction(i))
            dist = self.calculate_distance_to_end(newPosition)
            if(dist <= minDist):
                if(dist< minDist):
                    min = [i]
                else:
                    min.append(i)
                minDist = dist

        return Direction(random.choice(min))

    def get_next_move(self):
        if self.strategy == 1:
            return self.greedy_strategy()
        elif self.strategy == 2:
            return self.a_star_like()

    def get_max_value(self, position):
        return numpy.amax(self.get_q_row(position))

    def get_next_position(self, move_direction):
        if move_direction == Direction.TOP:
            return Position(self.currentPosition.x, self.currentPosition.y - 1)
        elif move_direction == Direction.RIGHT:
            return Position(self.currentPosition.x + 1, self.currentPosition.y)
        elif move_direction == Direction.DOWN:
            return Position(self.currentPosition.x, self.currentPosition.y + 1)
        elif move_direction == Direction.LEFT:
            return Position(self.currentPosition.x - 1, self.currentPosition.y)

    def make_next_move(self, move_direction):
        previous_position = self.currentPosition
        self.currentPosition = self.get_next_position(move_direction)
        reward = self.R[self.get_position(previous_position)][move_direction.value]
        maxNextQValue = self.get_max_value(self.get_position(self.currentPosition))
        currentValue = self.Q[self.get_position(previous_position)][move_direction.value]
        punishment = self.P[self.get_position(previous_position)][move_direction.value]
        self.Q[self.get_position(previous_position)][move_direction.value] = (1 - self.learningRate) * currentValue + self.learningRate * (
                                                                                         reward + self.futureStepsRate * maxNextQValue)
        self.P[self.get_position(previous_position)][move_direction.value] = punishment + 1
