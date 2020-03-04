import dxchange
import numpy as np
import deformcg as df
import elasticdeform
import matplotlib.pyplot as plt
import cv2

def deform(data):
    res = data.copy()
    points = [3,3]
    # displacement = (np.random.rand(2, *points) - 0.5)* 10
    #np.save('disp',displacement)
    displacement=np.load('disp.npy')
    for k in range(0,ntheta):
      res[k].real = elasticdeform.deform_grid(data[k].real,displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
      res[k].imag = elasticdeform.deform_grid(data[k].imag,displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
    return res

if __name__ == "__main__":

    # Model parameters
    n = 628  # object size n x,y
    nz = 128  # object size in z
    ntheta = 1  # number of angles (rotations)
    # Load object
    # beta = dxchange.read_tiff('data/beta-chip-256.tiff')[:,64:64+ntheta].swapaxes(0,1)
    # delta = dxchange.read_tiff('data/delta-chip-256.tiff')[:,64:64+ntheta].swapaxes(0,1)
    # u0 = delta+1j*beta
    # u = deform(u0) 
    # dxchange.write_tiff(u.real,'defdelta-chip-128.tiff',overwrite=True)
    # dxchange.write_tiff(u.imag,'defbeta-chip-128.tiff',overwrite=True)
    u0 = np.expand_dims(dxchange.read_tiff('data/battery0.tif'),0)+0j
    u = np.expand_dims(dxchange.read_tiff('data/battery180.tif'),0)+0j
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
    with df.SolverDeform(ntheta, nz, n) as slv:     
            flow = slv.registration_flow_batch(u0,u,flow,pars=[0.5, 1, 12, 16, 5, 1.1, 0])
            rec = slv.cg_deform(u,u*0,flow,12,dbg=False)        
            dxchange.write_tiff(rec.real,'data/abattery180',overwrite=True)        
            dxchange.write_tiff(rec.real-u0.real,'data/diffr',overwrite=True)        
            dxchange.write_tiff(u.real-u0.real,'data/diff0',overwrite=True)        
            plt.imshow(df.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')
            print(np.linalg.norm(rec.real-u0.real))
            plt.savefig('flow')
            plt.close()
