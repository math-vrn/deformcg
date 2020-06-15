import dxchange
import numpy as np
import deformcg as df
import elasticdeform
import matplotlib.pyplot as plt
import cv2

def deform(data):
    res = data.copy()
    points = [3,3]
    displacement = (np.random.rand(2, *points) - 0.5)* 5
    for k in range(0,ntheta):
      res[k] = elasticdeform.deform_grid(data[k].real,displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
    return res

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 1  # number of angles (rotations)
    ptheta = 1
    
    # Load object
    u0 = dxchange.read_tiff('data/delta-chip-128.tiff')[:,64:64+ntheta].swapaxes(0,1)

    # deform it    
    u = deform(u0) 
    
    mmin = np.min(u,axis=(1,2))
    mmax = np.max(u,axis=(1,2))
    
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    with df.SolverDeform(ntheta, nz, n, ptheta) as slv:     
            flow = slv.registration_flow_batch(u0,u,mmin,mmax,flow,pars=[0.5, 1, 16, 20, 5, 1.1, 0])
            rec = slv.cg_deform(u,u*0,flow,12,dbg=False)        
            
    dxchange.write_tiff(u0[ntheta//2],'res/delta-chip-128.tiff',overwrite=True)    
    dxchange.write_tiff(u[ntheta//2],'res/defdelta-chip-128.tiff',overwrite=True)
    dxchange.write_tiff(rec[ntheta//2],'res/recdefdelta-chip-128.tiff',overwrite=True)
    
    plt.imshow(df.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')
    plt.savefig('flow')
    plt.close()
