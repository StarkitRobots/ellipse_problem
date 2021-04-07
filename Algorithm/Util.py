import numpy as np
from .Point import Point


def create_points_from_numpyimage(arr_image:np.ndarray):
    lst=list()
    image_shape=arr_image.shape
    image_ht=image_shape[0]
    image_width=image_shape[1]
    for x in range(0,image_width):
        for y in range(0,image_ht):
            #print("x=%d y=%d" % (x,y))
            #Change coordinate system
            color=arr_image[y][x]
            #we want black pixels only
            if (color <=0.5):
                continue

            y_cartesian=image_ht - y -1
            p=(x,y_cartesian)
            lst.append(p)
    return lst
#
#Draws the specified collection of Point objects over the numpy array
#The coordinate system of the point will be transformed from Cartesian(bottom-left) to Image(top-left)
#
def superimpose_points_on_image(arr_image_input:np.ndarray, points,red:int,green:int,blue:int):
    width=arr_image_input.shape[1]
    height=arr_image_input.shape[0]
    arr_new=np.zeros([height,width,3])
    #We want to capture the original image
    for x in range(0,width):
        for y in range(0,height):
            color=arr_image_input[y][x]
            if (color[0] > 125 or color[1] > 125 or color[2] > 125  ):
                arr_new[y][x][0]=255
                arr_new[y][x][1]=255
                arr_new[y][x][2]=255
    #superimpose the points onto the numpy array
    for p in points:
        x:int=int(round(p.X))
        y:int=int(round(height-p.Y-1))
        if (x<0 or x >= width ):
            continue
        if (y<0 or y >= height ):
            continue
        arr_new[y][x][0]=red
        arr_new[y][x][1]=green
        arr_new[y][x][2]=blue
    return arr_new

