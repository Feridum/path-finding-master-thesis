import math
import random
import sys
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

MAX_TIME = 5*60
MAX_EPOCHES = 1000

class LearnRL:
    def __init__(self, maze, bonusFields=None, seed=None, R=None, Q=None, maxTime = None):
        if Q is None:
            Q = []
        if R is None:
            R = []
        if bonusFields is None:
            bonusFields = []

        if maxTime is None:
            maxTime = MAX_TIME

        sys.setrecursionlimit(2 * MAX_EPOCHES)
        self.start = numpy.where(maze == 'B')
        end = numpy.where(maze == 'E')
        self.startPosition = Position(self.start[1], self.start[0])
        self.maze = maze
        self.endPosition = Position(end[1], end[0])
        self.learningRate = 0.8
        self.futureStepsRate = 0.9
        self.strategy = 1
        self.rStrategy = 1
        self.R = R
        self.Q = Q
        self.N = []
        self.currentPosition = Position(0,0)
        self.stepsNumber = []
        self.maxDistance = 0
        self.bonusFields = bonusFields
        self.e = 0.1
        self.t = 0.5
        self.startTime = 0
        self.finishReason = 'COMPLETED'
        self.maxTime = maxTime
        random.seed(seed)

    def resetMaze(self):
        shape = self.maze.shape
        self.maxDistance = shape[0] + shape[1]
        self.R = numpy.zeros([shape[0] * shape[1], 4])
        self.Q = numpy.full([shape[0] * shape[1], 4], -math.inf, dtype=float)
        self.N = numpy.full([shape[0] * shape[1], 4], 1, dtype=float)
        # self.Q[self.get_position(self.endPosition)] = numpy.full(4, 0)
        self.currentPosition = Position(self.start[1], self.start[0])
        self.finishReason = 'COMPLETED'
        self.stepsNumber = []

    def startLearn(self, strategy = 1, rStrategy=1, learningRate = 0.8, futureStepsRate = 0.9, e=0.1, t=0.5):
        self.resetMaze()
        self.strategy = strategy
        self.rStrategy = rStrategy
        self.learningRate = learningRate
        self.futureStepsRate = futureStepsRate
        self.e = e
        self.t = t
        self.startTime = time.time()
        self.learn()
        end = time.time()
        return self.finishReason, end-self.startTime, len(self.stepsNumber), self.stepsNumber, self.Q, self.R

    def learn(self):
        self.singleEpoche()
        if time.time() - self.startTime > self.maxTime:
            print('max time reached')
            self.finishReason = 'MAX_TIME'
        elif self.stepsNumber.shape[0] > 3:
            if self.stepsNumber[-1] == self.stepsNumber[-2] and self.stepsNumber[-2] == self.stepsNumber[-3]:
                print('finished')
                self.finishReason = 'COMPLETED'
            elif len(self.stepsNumber) > MAX_EPOCHES:
                print('max epoches reached')
                self.finishReason = 'MAX_EPOCHE'
            else:
                self.currentPosition = self.startPosition
                self.learn()
        else:
            self.currentPosition = self.startPosition
            self.learn()

    def singleEpoche(self):
        steps_number = 0
        shape = self.maze.shape
        self.N = numpy.full([shape[0] * shape[1], 4], 1, dtype=float)
        while self.currentPosition != self.endPosition:
            self.discover_neighbourhood()
            self.make_next_move(self.get_next_move())
            steps_number = steps_number + 1
            if time.time() - self.startTime > self.maxTime:
                self.finishReason = 'MAX_TIME'
                break

        self.t = self.t - 0.001
        if(self.t < 0.1):
            self.t = 0.1

        self.stepsNumber = numpy.append(self.stepsNumber, steps_number)

    def startFindPath(self, Q, R,  strategy = 1, rStrategy=1, learningRate = 0.8, futureStepsRate = 0.9, e=0.1, t=0.5):
        self.resetMaze()
        self.strategy = strategy
        self.rStrategy = rStrategy
        self.learningRate = learningRate
        self.futureStepsRate = futureStepsRate
        self.e = e
        self.t = t
        self.Q = Q
        self.R = R
        self.startTime = time.time()
        self.singleEpoche()
        end = time.time()
        return end - self.startTime, self.stepsNumber[0], self.finishReason

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
            self.R[position][direction.value] = -0.1
        else:
            if self.rStrategy == 1:
                self.R[position][direction.value] = -1
            elif self.rStrategy == 2:
                dist = self.calculate_distance_to_end(self.get_next_position(direction))
                if dist <= 0.3 * self.maze.shape[0]:
                    self.R[position][direction.value] = -dist/self.maze.shape[0]
                else:
                    self.R[position][direction.value] = -1

    def get_q_row(self, position):
        q = numpy.copy(self.Q[position])
        for i in range(4):
            if self.R[position][i] == -math.inf:
                q[i] = -math.inf
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

    def e_greedy_strategy(self):
        position = self.get_position(self.currentPosition)
        values = numpy.argwhere(self.get_q_row(position) != -math.inf)
        if numpy.random.rand(1) < self.e:
            return Direction(random.choice(values))
        else:
            return self.greedy_strategy()

    def boltzman_strategy(self):
        position = self.get_position(self.currentPosition)
        qRow = self.get_q_row(position)
        values = numpy.argwhere(qRow != -math.inf).flatten()
        numerator = []
        p = []
        sum = 0;
        for v in values:
            val = ( qRow[v]/self.t )/100
            num = math.exp(val)
            sum = sum + num;
            numerator.append(num)
        for n in numerator:
            p.append(n/sum)

        return Direction(numpy.random.choice(values, p=p))

    def calculate_distance_to_end(self, position):
        return abs(position.x - self.endPosition.x) + abs(position.y - self.endPosition.y)

    def calculate_distance_to_end2(self, position):
        return math.sqrt(pow((position.x - self.endPosition.x),2) + pow((position.y - self.endPosition.y),2))

    def ucb(self):
        position = self.get_position(self.currentPosition)
        qRow = self.get_q_row(position)
        indicies = numpy.argwhere(self.get_q_row(position) != -math.inf)
        value = numpy.full(4, -math.inf, dtype=float)
        sum_n = numpy.sum(self.N[position])
        epoches = len(self.stepsNumber)
        for i in indicies:
            n = self.N[position][i]
            dist = self.calculate_distance_to_end(self.currentPosition)
            c = (dist/(2*self.maze.shape[0])) * (10 - (epoches//50)*0.5 + 0.5)

            value[i] = qRow[i] + ( c * math.log(2*sum_n/n))

        max_indices = numpy.argwhere(value == numpy.amax(value)).flatten()

        return Direction(random.choice(max_indices))

    def get_next_move(self):
        if self.strategy == 1:
            return self.greedy_strategy()
        elif self.strategy == 2:
            return self.e_greedy_strategy()
        elif self.strategy == 3:
            return self.boltzman_strategy()
        elif self.strategy == 4:
            return self.ucb()

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

        self.N[self.get_position(previous_position)][move_direction.value] = self.N[self.get_position(previous_position)][move_direction.value] + 1
        if currentValue == -math.inf:
            currentValue = 0

        if(maxNextQValue == -math.inf):
            maxNextQValue = 0

        newValues = (reward + self.futureStepsRate * maxNextQValue)
        self.Q[self.get_position(previous_position)][move_direction.value] = (1 - self.learningRate) * currentValue + self.learningRate * newValues
