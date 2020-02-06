import time

from numpy import load

from AStar.learnAStar import LearnAStar
from displayPath import visualizeRL
from learnRL import LearnRL
from readMaze import parseMaze


def start_experiment():
    # maze = parseMaze('./10_simple.txt')
    maze = parseMaze('./maps/txt/map1.txt')
    learnRL = LearnRL(maze)
    print(learnRL.startLearn(2))
    # print(learnRL.startFindPath())
    # print(learnRL.startFindPath())

def start_a_star_experiment():
    # maze = parseMaze('./10_simple.txt')
    maze = parseMaze('./maps/txt/map1.txt')
    learn_a_star = LearnAStar(maze)
    print(learn_a_star.start())
    print(learn_a_star.start())
    print(learn_a_star.start())
    print(learn_a_star.start())
    print(learn_a_star.start())

def image():
    visualizeRL(load('./Q.npy'),100,100)

if __name__ == "__main__":
    start_experiment()
    # start_a_star_experiment()
    # image()