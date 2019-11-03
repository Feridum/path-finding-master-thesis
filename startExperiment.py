import time

import numpy

from AStar.learnAStar import LearnAStar
from learnRL import LearnRL
from readMaze import parseMaze


def start_experiment():
    maze = parseMaze('./10_simple.txt')
    learnRL = LearnRL(maze)
    start = time.time()
    learnRL.learn()
    end = time.time()
    print(end - start)

def start_a_star_experiment():
    maze = parseMaze('./10_simple.txt')
    learn_a_star = LearnAStar(maze)
    learn_a_star.start()

if __name__ == "__main__":
    start_a_star_experiment()