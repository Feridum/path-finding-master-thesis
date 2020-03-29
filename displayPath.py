import math

from PIL import Image
import numpy as np

def visualizeRL(Q, width,height):
    wall = Image.new("RGB", (64, 64), "#000000")
    finish = Image.new("RGB", (64, 64), "#FF0000")
    arrow = Image.open('./arrow.png')
    im = Image.new('RGB', (width*64, height*64))
    max = np.amax(Q)

    x_offset = 0
    y_offset = 0
    for h in range(0,height):
        for w in range(0,width):
            index = h*width + w
            row = Q[index]
            maxIndex = np.argmax(row)
            max = np.max(row)
            if max == -math.inf:
                im.paste(wall, (x_offset, y_offset))
            elif max == 1000:
                im.paste(finish, (x_offset, y_offset))
            else:
                if maxIndex == 0:
                    im.paste(arrow.rotate(90), (x_offset, y_offset))
                elif maxIndex == 1:
                    im.paste(arrow, (x_offset, y_offset))
                elif maxIndex == 2:
                    im.paste(arrow.rotate(-90), (x_offset, y_offset))
                elif maxIndex == 3:
                    im.paste(arrow.rotate(180), (x_offset, y_offset))

            x_offset+=64
        x_offset = 0
        y_offset+=64

    im.show()

def visualizeAStar(maze, path, closeList):
    height = maze.shape[0]
    width = maze.shape[1]
    wall = Image.new("RGB", (64, 64), "#000000")
    empty = Image.new("RGB", (64, 64), "#FFFFFF")
    start = Image.new("RGB", (64, 64), "#00FF00")
    finish = Image.new("RGB", (64, 64), "#FF0000")
    pathImage = Image.new("RGB", (64, 64), "#0000FF")
    visited = Image.new("RGB", (64, 64), "#87CEEB")

    im = Image.new('RGB', (width * 64, height * 64))

    x_offset = 0
    y_offset = 0
    for h in range(0, height):
        for w in range(0, width):
            if maze[h][w] == '#':
                im.paste(wall, (x_offset, y_offset))
            elif maze[h][w] == 'B':
                im.paste(start, (x_offset, y_offset))
            elif maze[h][w] == 'E':
                im.paste(finish, (x_offset, y_offset))
            elif get_position(w,h,width) in path:
                im.paste(pathImage, (x_offset, y_offset))
            elif get_position(w,h,width) in closeList:
                im.paste(visited, (x_offset, y_offset))
            else:
                im.paste(empty, (x_offset, y_offset))
            x_offset += 64
        x_offset = 0
        y_offset += 64

    return im

def get_position(x,y, width):
    return y * width + x