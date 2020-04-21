import dxchange
import numpy as np
import deformcg as df
import matplotlib.pyplot as plt
from timing import *

if __name__ == "__main__":

    # Model parameters
    n = 648  # object size n x,y
    nz = 128  # object size in z
    ntheta = 256  # number of angles (rotations)
    # Load object
    u = np.zeros([ntheta,nz,n],dtype='float32')
    u = np.tile(dxchange.read_tiff('data/battery180.tif'),(ntheta,1,1))
    #u[1] = dxchange.read_tiff('data/battery0.tif')
    
    shifts = np.random.random([ntheta,2]).astype('float32')*16
    with df.SolverDeform(ntheta, nz, n) as slv:     

            tic()
            us = slv.apply_shift(u,shifts)
            print(toc())            
            tic()            
            shifts2 = slv.registration_shift(us,u,upsample_factor=4,space='real')
            print(toc())            
            print(cp.linalg.norm(shifts-shifts2))
            