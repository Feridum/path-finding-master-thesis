from numpy import save, asarray
import pathlib

from AStar.learnAStar import LearnAStar
from BFS.learnBFS import LearnBFS
from Dijkstra.learnDijkstra import LearnDijkstra
from createResultCsv import createSummaryResult
from displayPath import visualizeAStar
from readMaze import parseMaze

RESULT_PATH = 'results/data/'
IMAGE_PATH = 'results/images/'

def runSingle(mazeName):
    pathlib.Path(RESULT_PATH + mazeName).mkdir(parents=True, exist_ok=True)
    pathlib.Path(IMAGE_PATH + mazeName).mkdir(parents=True, exist_ok=True)

    maze = parseMaze('./maps/txt/'+ mazeName + '.txt')
    learn_a_star = LearnAStar(maze)
    learn_dijkstra = LearnDijkstra(maze)
    learn_bfs = LearnBFS(maze)

    a_results = learn_a_star.start()
    d_results = learn_dijkstra.start()
    b_results = learn_bfs.start()

    save(RESULT_PATH+mazeName+'/a_star.npy', asarray(a_results))
    save(RESULT_PATH+mazeName+'/dijkstra.npy', asarray(d_results))
    save(RESULT_PATH+mazeName+'/bfs.npy', asarray(b_results))

    a_i = visualizeAStar(maze, a_results[2], a_results[3])
    d_i = visualizeAStar(maze, d_results[2], d_results[3])
    b_i = visualizeAStar(maze, b_results[2], b_results[3])

    a_i.save(IMAGE_PATH+mazeName+'/a_star.png')
    d_i.save(IMAGE_PATH+mazeName+'/dijkstra.png')
    b_i.save(IMAGE_PATH+mazeName+'/bfs.png')

    createSummaryResult(mazeName, ['a_star', 'dijkstra', 'bfs'])

def runAll(mazeType):
    sizes = [10, 50, 100]
    variants = ['', '_change_1', '_change_2', '_change_start']

    for s in sizes:
        for v in variants:
            filename = str(s) + 'x' + str(s) + '_' + mazeType + v
            print('<----- Start' + filename + '------>')
            runSingle(filename)
            print('<----- End' + filename + '------>')

if __name__ == "__main__":
    mazes = ['wall', 'columns', 'board']
    for m in mazes:
        runAll(m)