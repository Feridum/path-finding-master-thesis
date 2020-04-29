from PIL import Image
import numpy as np;

IMAGE_PATH = 'maps/png/'
TXT_PATH = 'maps/txt/'
EMPTY = '.'
WALL = '#'
BEGIN = 'B'
END = 'E'

def createMaze(filename):
    im = Image.open(IMAGE_PATH + filename + '.png')
    pix = im.load()
    width, height = im.size
    result = np.empty((width, height), dtype=str)

    for y in range(0, height):
        for x in range(0, width):
            if(pix[x,y] == (255, 255, 255, 255)):
                result[y,x] = EMPTY
            elif (pix[x, y] == (0, 0, 0, 0)):
                result[y, x] = EMPTY
            elif(pix[x,y] == (0, 0, 0, 255)):
                result[y,x] = WALL
            elif(pix[x,y] == (255, 0, 0, 255)):
                result[y,x] = END
            elif(pix[x,y] == (0, 255, 0, 255)):
                result[y,x] = BEGIN

    with open(TXT_PATH+filename+'.txt', "w") as txt_file:
        txt_file.write("" + str(height) + " " + str(width) + "\n")
        for line in result:
            txt_file.write("".join(line) + "\n")

def createMazeForMazeType(mazeType):
    sizes = [10,50,100]
    variants = ['', '_change_1', '_change_2', '_change_start']

    for s in sizes:
        for v in variants:
            createMaze(str(s)+'x'+str(s)+'_'+mazeType+v)

if __name__ == "__main__":
    mazes = ['wall', 'columns', 'board']
    for m in mazes:
        createMazeForMazeType(m)