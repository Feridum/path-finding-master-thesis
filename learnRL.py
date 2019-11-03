import random

import numpy
from collections import namedtuple
from enum import Enum

Position = namedtuple('Position', 'x y')


class Direction(Enum):
    TOP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class LearnRL:
    def __init__(self, maze):
        start = numpy.where(maze == 'B')
        end = numpy.where(maze == 'E')
        shape = maze.shape
        self.R = numpy.zeros([shape[0] * shape[1], 4])
        self.Q = numpy.ones([shape[0] * shape[1], 4])
        self.startPosition = Position(start[1], start[0])
        self.maze = maze
        self.currentPosition = Position(start[1], start[0])
        self.endPosition = Position(end[1], end[0])
        self.learningRate = 1
        self.futureStepsRate = 0.6
        self.stepsNumber = []

    def learn(self):
        steps_number = 0
        while self.currentPosition != self.endPosition:
            self.discover_neighbourhood()
            self.make_next_move(self.get_next_move())
            steps_number = steps_number + 1
        self.stepsNumber = numpy.append(self.stepsNumber, steps_number)
        if self.stepsNumber.shape[0] > 3:
            if self.stepsNumber[-1] == self.stepsNumber[-2] and self.stepsNumber[-2] == self.stepsNumber[-3]:
                print(self.Q)
            else:
                self.currentPosition = self.startPosition
                self.learn()
        else:
            self.currentPosition = self.startPosition
            self.learn()

    def discover_neighbourhood(self):
        x = self.currentPosition.x[0]
        y = self.currentPosition.y[0]
        position = self.get_position(self.currentPosition)

        self.R[position] = numpy.full(4, -100)

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
            self.R[position][direction.value] = -100
        elif self.maze[y][x] == 'E':
            self.R[position][direction.value] = 100
        else:
            self.R[position][direction.value] = 1

    def get_q_row(self, position):
        q = self.Q[position]
        for i in range(4):
            if self.R[position][i] == -100:
                q[i] = -100
            elif self.R[position][i] == 100 and self.Q[position][i] == 0:
                q[i] = 100
        return q

    def get_position(self, position):
        x = position.x[0]
        y = position.y[0]
        return y * self.maze.shape[0] + x

    def get_next_move(self):
        position = self.get_position(self.currentPosition)
        max_indices = numpy.argwhere(self.get_q_row(position) == self.get_max_value(position))
        return Direction(random.choice(max_indices))

    def get_max_value(self, position):
        return numpy.amax(self.get_q_row(position))

    def make_next_move(self, move_direction):
        previous_position = self.currentPosition
        if move_direction == Direction.TOP:
            self.currentPosition = Position(self.currentPosition.x, self.currentPosition.y - 1)
        elif move_direction == Direction.RIGHT:
            self.currentPosition = Position(self.currentPosition.x + 1, self.currentPosition.y)
        elif move_direction == Direction.DOWN:
            self.currentPosition = Position(self.currentPosition.x, self.currentPosition.y + 1)
        elif move_direction == Direction.LEFT:
            self.currentPosition = Position(self.currentPosition.x - 1, self.currentPosition.y)

        reward = self.R[self.get_position(previous_position)][move_direction.value]
        maxNextQValue = self.get_max_value(self.get_position(self.currentPosition))
        self.Q[self.get_position(previous_position)][move_direction.value] = (1 - self.learningRate) * self.Q[
            self.get_position(previous_position)][move_direction.value] + self.learningRate * (
                                                                                         reward + self.futureStepsRate * maxNextQValue) - 1;
