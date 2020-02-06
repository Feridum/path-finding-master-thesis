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


if __name__ == "__main__":
    createMaze('map1')