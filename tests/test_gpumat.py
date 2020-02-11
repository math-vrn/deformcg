from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import matplotlib.pyplot as plt
## [Update]
def update_map(ind, map_x, map_y):
    if ind == 0:
        for i in range(map_x.shape[0]):
            for j in range(map_x.shape[1]):
                if j > map_x.shape[1]*0.25 and j < map_x.shape[1]*0.75 and i > map_x.shape[0]*0.25 and i < map_x.shape[0]*0.75:
                    map_x[i,j] = 2 * (j-map_x.shape[1]*0.25) + 0.5
                    map_y[i,j] = 2 * (i-map_y.shape[0]*0.25) + 0.5
                else:
                    map_x[i,j] = 0
                    map_y[i,j] = 0
    elif ind == 1:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
    elif ind == 2:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [y for y in range(map_y.shape[0])]
    elif ind == 3:
        for i in range(map_x.shape[0]):
            map_x[i,:] = [map_x.shape[1]-x for x in range(map_x.shape[1])]
        for j in range(map_y.shape[1]):
            map_y[:,j] = [map_y.shape[0]-y for y in range(map_y.shape[0])]
## [Update]

parser = argparse.ArgumentParser(description='Code for Remapping tutorial.')
parser.add_argument('--input', help='Path to input image.', default='res.png')
args = parser.parse_args()

## [Load]
src = cv.imread(cv.samples.findFile(args.input), cv.IMREAD_COLOR)
if src is None:
    print('Could not open or find the image: ', args.input)
    exit(0)
## [Load]

## [Create]
map_x = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
map_y = np.zeros((src.shape[0], src.shape[1]), dtype=np.float32)
## [Create]

## [Window]
window_name = 'Remap demo'
#cv.namedWindow(window_name)
## [Window]

## [Loop]
ind = 0
k=0
while True:
    update_map(ind, map_x, map_y)
    ind = (ind + 1) % 4
    srcgpu = cv.cuda_GpuMat()
    map_xgpu = cv.cuda_GpuMat()
    map_ygpu = cv.cuda_GpuMat()
    srcgpu.upload(src)    
    map_xgpu.upload(map_x)    
    map_ygpu.upload(map_y)    
    print('start')
    dstgpu = cv.cuda.remap(srcgpu, map_xgpu, map_ygpu, cv.INTER_LINEAR)
    print('end')    
    plt.imshow(dstgpu)

    plt.savefig(str(k)+'.png')    
    k=k+1
    print(k)
## [Loop]