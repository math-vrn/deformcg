import dxchange
import numpy as np
import deformcg as df
import elasticdeform
import matplotlib.pyplot as plt
import os 

def myplot(u0, u, flow):
    [ntheta, nz, n] = u.shape

    plt.figure(figsize=(20, 14))
    plt.subplot(1, 3, 1)
    plt.imshow(u0[0].real, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.imshow(u[0].real, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.imshow(df.flowvis.flow_to_color(flow[0]), cmap='gray')
    if not os.path.exists('tmp'+'_'+str(ntheta)+'/'):
        os.makedirs('tmp'+'_'+str(ntheta)+'/')
    plt.savefig('tmp'+'_'+str(ntheta)+'/flow')
    plt.close()
    print(np.linalg.norm(flow))        

def deform(data):
    res = data.copy()
    points = [3, 3]
    displacement = (np.random.rand(2, *points) - 0.5)* 10
#     np.save('disp',displacement)
#    displacement = np.load('disp.npy')
    for k in range(0, ntheta):
        res.real = elasticdeform.deform_grid(
            data[k].real, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
        res.imag = elasticdeform.deform_grid(
            data[k].imag, displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
    return res


if __name__ == "__main__":

      # Model parameters
      n = 128  # object size n x,y
      nz = 128  # object size in z
      ntheta = 16  # number of angles (rotations)
      # Load object
      beta = dxchange.read_tiff(
            'data/beta-chip-128.tiff')[:, 64:64+ntheta].swapaxes(0, 1)
      delta = dxchange.read_tiff(
            'data/delta-chip-128.tiff')[:, 64:64+ntheta].swapaxes(0, 1)
      u0 = delta+1j*beta*0
      pars=[4, 0.5, False, 23, 10, 5, 1.1, 4]
      flow = np.zeros([ntheta, nz, n, 2], dtype='float32')
      #int numLevels=5, double pyrScale=0.5, bool fastPyramids=false, int winSize=13, int numIters=10, int polyN=5, double polySigma=1.1, int flags=0
      with df.SolverDeform(ntheta, nz, n) as slv:
            u = deform(u0)
            flow = slv.registration_flow_batch(u0, u, flow, pars).astype('double')
            myplot(u0,u,flow)
            u1 = slv.apply_flow_batch(u0, flow)
            u2 = slv.apply_flow_batch(u1, -flow)
            print('Adjoint test optical flow: ', np.sum(u1*np.conj(u1)),
                  '=?', np.sum(u0*np.conj(u2)))
            plt.subplot(2, 3, 1)
            plt.imshow(u[0].real, cmap='gray')
            plt.colorbar()
            plt.subplot(2, 3, 2)
            plt.imshow(u1[0].real, cmap='gray')
            plt.colorbar()
            plt.subplot(2, 3, 3)
            plt.imshow(u[0].real-u1[0].real, cmap='gray')
            plt.colorbar()
            plt.subplot(2, 3, 4)
            plt.imshow(u0[0].real, cmap='gray')
            plt.colorbar()
            plt.subplot(2, 3, 5)
            plt.imshow(u2[0].real, cmap='gray')
            plt.colorbar()
            plt.subplot(2, 3, 6)
            plt.imshow(u0[0].real-u2[0].real, cmap='gray')
            plt.colorbar()
            plt.savefig('res.png',dpi=1000)
            print(np.linalg.norm(u[0].real-u1[0].real))
            print(np.linalg.norm(u0[0].real-u2[0].real))            
            exit()
            shift0 = np.random.random([ntheta, 2])
            u = slv.apply_shift_batch(u0, shift0)
            shift = slv.registration_shift_batch(u, u0, 10)
            print('Check shift:', np.linalg.norm(shift-shift0))
            print(shift[0], '<->', shift0[0])
            print(shift[ntheta-1], '<->', shift0[ntheta-1])
            u1 = slv.apply_shift_batch(u0, shift)
            u2 = slv.apply_shift_batch(u1, -shift)
            print('Adjoint test shift: ', np.sum(u1*np.conj(u1)),
                  '=?', np.sum(u0*np.conj(u2)))
            print('Inverse test shift: ', np.linalg.norm(u0-u2)/np.linalg.norm(u0))
