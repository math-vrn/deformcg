import dxchange
import numpy as np
import deformcg as df
import elasticdeform

def deform(data):
    res = data.copy()
    points = [3,3]
    displacement = (np.random.rand(2, *points) - 0.5)* 10
    for k in range(0,ntheta):
      res[k].real = elasticdeform.deform_grid(data[k].real,displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
      res[k].imag = elasticdeform.deform_grid(data[k].imag,displacement,order=5,mode='mirror',crop=None,prefilter=True,axis=None)                                    
    return res

if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 1  # number of angles (rotations)
    # Load object
    beta = dxchange.read_tiff('data/beta-chip-128.tiff')[:,64:64+ntheta].swapaxes(0,1)
    delta = dxchange.read_tiff('data/delta-chip-128.tiff')[:,64:64+ntheta].swapaxes(0,1)
    u0 = delta+1j*beta
    u = deform(u0)    
    dxchange.write_tiff(u.real,'defdelta-chip-128.tiff',overwrite=True)
    dxchange.write_tiff(u.imag,'defbeta-chip-128.tiff',overwrite=True)
    with df.SolverDeform(ntheta, nz, n) as slv:     
        flow = slv.registration_flow_batch(u0,u)           
        rec = slv.cg_deform(u,u*0,flow,12,dbg=True)        
        dxchange.write_tiff(rec.real,'recdelta-chip-128.tiff',overwrite=True)
        dxchange.write_tiff(rec.imag,'recbeta-chip-128.tiff',overwrite=True)