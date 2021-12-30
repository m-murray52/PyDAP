import cv2
import numpy as np
import argparse

def get8n(x, y, shape):
    out = []
    maxx = shape[1]-1
    maxy = shape[0]-1
    
    #top left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top center
    outx = x
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #top right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y-1,0),maxy)
    out.append((outx,outy))
    
    #left
    outx = min(max(x-1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #right
    outx = min(max(x+1,0),maxx)
    outy = y
    out.append((outx,outy))
    
    #bottom left
    outx = min(max(x-1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom center
    outx = x
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    #bottom right
    outx = min(max(x+1,0),maxx)
    outy = min(max(y+1,0),maxy)
    out.append((outx,outy))
    
    return out



class RegionGrow:
    
    #clicks = []
    
    def __init__(self) -> None:
        pass


    def region_grower(img, seed) -> tuple:
        seed_points = []
        outimg = np.zeros_like(img)
        seed_points.append((seed[0], seed[1]))
        processed = []
        while(len(seed_points) > 0):
            pix = seed_points[0]
            outimg[pix[0], pix[1]] = 255
            for coord in get8n(pix[0], pix[1], img.shape):
                if img[coord[0], coord[1]] != 0:
                    outimg[coord[0], coord[1]] = 255
                    if not coord in processed:
                        seed_points.append(coord)
                    processed.append(coord)
            seed_points.pop(0)
        
        return outimg

    #def on_mouse(self, event, x, y, flags, params):
    #    if event == cv2.EVENT_LBUTTONDOWN:
    #        print('Seed: ' + str(x) + ', ' + str(y), self.img[y,x])
    #        self.clicks.append((y,x))

    #cv2.namedWindow('Input')
    #cv2.setMouseCallback('Input', on_mouse, 0, )
    #cv2.imshow('Input', img)
    #cv2.waitKey()
    #seed = clicks[-1]
    #out = region_grower(img, seed)
    #cv2.imshow('Region Growing', out)

class Bag:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def addtwice(self, x):
        self.add(x)
        self.add(x)