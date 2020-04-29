from numpy import load,save,asarray

from AStar.learnAStar import LearnAStar
from BFS.learnBFS import LearnBFS
from Dijkstra.learnDijkstra import LearnDijkstra
from displayPath import visualizeRL, visualizeAStar
from RL.learnRL import LearnRL
from readMaze import parseMaze


def start_experiment():
    # maze = parseMaze('./10_simple.txt')
    maze = parseMaze('./maps/txt/50x50_wall.txt')
    # learn_a_star = LearnAStar(maze)
    # aResults = learn_a_star.start()
    # print(aResults[2])
    learnRL = LearnRL(maze, seed=1584259720496)
    results = learnRL.startLearn(strategy=4, rStrategy=1)
    print(results)
    # visualizeRL(results[4], 10,10)
    # save('results2', asarray(results))

def start_a_star_experiment():
    maze = parseMaze('./maps/txt/50x50_wall.txt')
    # maze = parseMaze('./maps/txt/map1.txt')
    learn_a_star = LearnAStar(maze)
    [time, length, path, visited] = learn_a_star.start()
    # visualizeAStar(maze, path, visited)
    save('resultastar', asarray([time,length,path,visited]))


def start_dijkstra_experiment():
    # maze = parseMaze('./10_simple.txt')
    maze = parseMaze('./maps/txt/map1.txt')
    learn_dijkstra = LearnDijkstra(maze)
    [time, length, path, visited]  = learn_dijkstra.start()
    visualizeAStar(maze, path, visited)

def start_bfs_experiment():
    maze = parseMaze('./10_simple.txt')
    # maze = parseMaze('./maps/txt/map1.txt')
    learn_bfs = LearnBFS(maze)
    [time, length, path, visited]  = learn_bfs.start()
    print(length, time, path)
    visualizeAStar(maze, path, visited)

def image():
    visualizeRL(load('./Q.npy'),100,100)

if __name__ == "__main__":
    # start_experiment()
    start_a_star_experiment()
    # start_dijkstra_experiment()
    # start_bfs_experiment()
    # image()