import dxchange
import numpy as np
import deformcg as dc
import tomocg as tc
import elasticdeform
import matplotlib.pyplot as plt
import os 
from scipy import ndimage

def myplot(u0, u, flow):
    [ntheta, nz, n] = u.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(u0[0].real, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(u[0].real, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.imshow(dc.flowvis.flow_to_color(flow[0]), cmap='gray')
    if not os.path.exists('tmp'+'_'+str(ntheta)+'/'):
        os.makedirs('tmp'+'_'+str(ntheta)+'/')
    plt.savefig('tmp'+'_'+str(ntheta)+'/flow')
    plt.close()
    print(np.linalg.norm(flow))        

def deform(data):
    res = data.copy()
    points = [3, 3, 3]
    displacement = (np.random.rand(3, *points) - 0.5)* 10
#     np.save('disp',displacement)
 #   displacement = np.load('disp.npy')
    res.real = elasticdeform.deform_grid(
            data.real, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
    res.imag = elasticdeform.deform_grid(
            data.imag, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
    return res


if __name__ == "__main__":

      # Model parameters
      n = 128  # object size n x,y
      nz = 128  # object size in z
      ntheta = 1  # number of angles (rotations)
      theta = np.linspace(0, np.pi, ntheta).astype('float32')  # angles
      # Load object
      beta = dxchange.read_tiff(
            'data/beta-chip-128.tiff')#[:, 64:64+ntheta].swapaxes(0, 1)
      delta = dxchange.read_tiff(
            'data/delta-chip-128.tiff')#[:, 64:64+ntheta].swapaxes(0, 1)
      u0 = delta+1j*beta*0
      pars=[4, 0.5, False, 8, 16, 7, 1.1, 4]
      flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
      #int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0
      with dc.SolverDeform(ntheta, nz, n) as dslv:
            with tc.SolverTomo(theta, ntheta, nz, n, 128, 64) as tslv:
                  # generate data
                  psi0 = tslv.fwd_tomo_batch(u0)        
                  #u = deform(u0)
                  #np.save('u',u)
                  u = np.load('u.npy')
                  psi = tslv.fwd_tomo_batch(u)        
                  #dxchange.write_tiff_stack(u.real,'u/u.tiff')
                  #dxchange.write_tiff(psi.real,'psi/psi.tiff')
                  flow = dslv.registration_flow_batch(psi0, psi, flow, pars).astype('double')
                  #flow = np.random.random(flow.shape)*4
                  myplot(psi0,psi,flow)
                  psi1 = dslv.apply_flow_batch(psi0, flow)
                  psi2 = dslv.apply_flow_batch(psi1, -flow)
                  print('Adjoint test optical flow: ', np.sum(psi1*np.conj(psi1)),
                        '=?', np.sum(psi0*np.conj(psi2)))
                  plt.subplot(2, 3, 1)
                  plt.imshow(psi[0].real, cmap='gray')
                  plt.colorbar()
                  plt.subplot(2, 3, 2)
                  plt.imshow(psi1[0].real, cmap='gray')
                  plt.colorbar()
                  plt.subplot(2, 3, 3)
                  plt.imshow(psi[0].real-psi1[0].real, cmap='gray')
                  plt.colorbar()
                  plt.subplot(2, 3, 4)
                  plt.imshow(psi0[0].real, cmap='gray')
                  plt.colorbar()
                  plt.subplot(2, 3, 5)
                  plt.imshow(psi2[0].real, cmap='gray')
                  plt.colorbar()
                  plt.subplot(2, 3, 6)
                  plt.imshow(psi0[0].real-psi2[0].real, cmap='gray')
                  plt.colorbar()
                  plt.savefig('res.png',dpi=600)
                  #plt.show()                              
                  