import dxchange
import numpy as np
import deformcg as df
import elasticdeform
import matplotlib.pyplot as plt
import cv2
from timing import tic,toc
def deform(data):
    res = data.copy()
    points = [3,3]
    displacement = (np.random.rand(2, *points) - 0.5)* 10
    print(data.shape)
    for k in range(0,ntheta):
      res[k] = elasticdeform.deform_grid(data[k],displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
    return res

if __name__ == "__main__":

    # Model parameters
    n = 648  # object size n x,y
    nz = 648  # object size in z
    ntheta = 2000  # number of angles (rotations)
    ptheta = 200

    u0 = np.zeros([ntheta,nz,n],dtype='float32')
    u0[:,:128,:648] = dxchange.read_tiff('data/battery0.tif')[:nz,:n].astype('float32')
    u0/=np.max(np.abs(u0))
    #u0 =np.expand_dims(u0,0)/np.max(np.abs(u0))
    #u0 = np.array(u0[:,:,:n],order='C')
    u=u0;# = deform(u0)
    mmin=np.min(u0)
    mmax=np.max(u0)
    
    flow = np.zeros([ntheta, nz, n, 2], dtype='float32')+1
    with df.SolverDeform(ntheta, nz, n, ptheta) as slv:     
            
            
            
            #flow = slv.registration_flow_batch(u0,u,mmin,mmax,flow,pars=[0.5, 1, 12, 16, 5, 1.1, 0])#*0
            #flow[:,10:,:,0]=10
            res = np.zeros([3,*u.shape[1:]],dtype='float32')
            res[0] = u0[0]
            tic()
            res[1] = slv.apply_flow_batch(u0,flow)[0]
            t1=toc()
            tic()
            res[2] = slv.apply_flow_gpu_batch(u0,flow)[0]
            t2=toc()
            print(t1,t2)
            print(res.shape)
            print(np.linalg.norm(res[1]-res[2]))
            # rec = slv.cg_deform(u,u*0,flow,12,dbg=False)        
            dxchange.write_tiff_stack(res,'data/res/res',overwrite=True)       
            # dxchange.write_tiff(res2,'data/res2',overwrite=True)        
            # dxchange.write_tiff(rec.real-u0.real,'data/diffr',overwrite=True)        
            # dxchange.write_tiff(u.real-u0.real,'data/diff0',overwrite=True)        
            # plt.imshow(df.flowvis.flow_to_color(flow[ntheta//2]), cmap='gray')
            # print(np.linalg.norm(rec.real-u0.real))
            # plt.savefig('flow')
            # plt.close()
