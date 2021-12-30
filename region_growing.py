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
    
    def __init__(self, img, seed) -> np.ndarray:
        self.seed_points = []
        
        self.outimg = np.zeros_like(img)
        self.seed_points.append((seed[0], seed[1]))
        self.processed = []
        while(len(self.seed_points) > 0):
            self.pix = self.seed_points[0]
            self.outimg[self.pix[0], self.pix[1]] = 255
            for coord in get8n(self.pix[0], self.pix[1], img.shape):
                if img[coord[0], coord[1]] != 0:
                    self.outimg[coord[0], coord[1]] = 255
                    if not coord in self.processed:
                        self.seed_points.append(coord)
                    self.processed.append(coord)
            self.seed_points.pop(0)
        return self.outimg

    def on_mouse(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print('Seed: ' + str(x) + ', ' + str(y), self.img[y,x])
            self.clicks.append((y,x))

    #cv2.namedWindow('Input')
    #cv2.setMouseCallback('Input', on_mouse, 0, )
    #cv2.imshow('Input', img)
    #cv2.waitKey()
    #seed = clicks[-1]
    #out = region_grower(img, seed)
    #cv2.imshow('Region Growing', out)
#print(RegionGrow.__dict__)

class Bag:
    def __init__(self):
        self.data = []

    def add(self, x):
        self.data.append(x)

    def addtwice(self, x):
        self.add(x)
        self.add(x)