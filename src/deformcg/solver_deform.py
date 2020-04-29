import numpy as np
import concurrent.futures as cf
import threading
from multiprocessing import Pool
from scipy import ndimage
from itertools import repeat
from functools import partial
from deformcg.deform import deform
from skimage.feature import register_translation
import cv2
import cupy as cp

class SolverDeform(deform):
    """Base class for deformation solvers.
    This class is a context manager which provides the basic operators required
    to implement a tomography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    ntheta : int
        The number of projections.    
    n, nz : int
        The pixel width and height of the projection.   
    """

    def __init__(self, ntheta, nz, n, ptheta):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, nz, n, ptheta)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def apply_flow_cpu(self, f, flow, id):
        """Apply optical flow for one projection."""
        flow0 = flow[id].copy()
        h, w = flow0.shape[:2]
        flow0 = -flow0
        flow0[:, :, 0] += np.arange(w)
        flow0[:, :, 1] += np.arange(h)[:, np.newaxis]
        f0 = f[id]
        res = cv2.remap(f0, flow0,
                                None, cv2.INTER_LANCZOS4)
        return res

    def apply_flow_cpu_batch(self, psi, flow,nproc=16):
        """Apply optical flow for all projection in parallel."""
        res = np.zeros(psi.shape, dtype='float32')
        with cf.ThreadPoolExecutor(nproc) as e:
            shift = 0
            for res0 in e.map(partial(self.apply_flow_cpu, psi, flow), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def registration_flow(self, psi, g, mmin,mmax, flow, pars, id):
        """Find optical flow for one projection"""
        tmp1 = ((psi[id]-mmin[id]) /
                        (mmax[id]-mmin[id])*255)
        tmp1[tmp1>255] = 255
        tmp1[tmp1<0] = 0
        tmp2 = ((g[id]-mmin[id]) /
                        (mmax[id]-mmin[id])*255)
        tmp2[tmp2>255] = 255
        tmp2[tmp2<0] = 0
        flow[id] = cv2.calcOpticalFlowFarneback(
            tmp1,tmp2, flow[id], *pars)
          
        return flow[id]

    def registration_flow_batch(self, psi, g, mmin,mmax, flow=None, pars=[0.5, 3, 20, 16, 5, 1.1, 4],nproc=16):
        """Find optical flow for all projections in parallel"""
        if (flow is None):
            flow = np.zeros([self.ntheta, self.nz, self.n, 2], dtype='float32')
        res = np.zeros([*psi.shape, 2], dtype='float32')
        with cf.ThreadPoolExecutor(nproc) as e:
            shift = 0
            for res0 in e.map(partial(self.registration_flow, psi, g, mmin,mmax,flow, pars), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def line_search(self, minf, gamma, psi, Tpsi, d, Td):
        """Line search for the step sizes gamma"""
        while(minf(psi, Tpsi)-minf(psi+gamma*d, Tpsi+gamma*Td) < 0):
            gamma *= 0.5
        return gamma
    
    def cg_deform(self, data, psi, flow, titer, xi1=0, rho=0, nproc=16, dbg=False):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi, Tpsi):
            f = np.linalg.norm(Tpsi-data)**2+rho*np.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Tpsi = self.apply_flow_cpu_batch(psi, flow,nproc)
            grad = (self.apply_flow_cpu_batch(Tpsi-data, -flow,nproc) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_flow_cpu_batch(d, flow,nproc)
            gamma = 0.5*self.line_search(minf, 1, psi,Tpsi,d,Td)
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            if (dbg and np.mod(i, 4) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi



    def apply_flow_gpu(self,f,flow):
       # g = cp.zeros([self.ptheta,self.nz,self.n],dtype='float32')
        h, w = flow.shape[1:3]
        flow = -flow.copy()
        flow[:,:, :, 0] += cp.arange(w)
        flow[:,:, :, 1] += cp.arange(h)[:, cp.newaxis]

        flowx = cp.asarray(flow[:,:,:,0],order='C')
        flowy = cp.asarray(flow[:,:,:,1],order='C')
        g = f.copy()# keep values that were not affected
        self.remap(g.data.ptr,f.data.ptr,flowx.data.ptr,flowy.data.ptr)
        return g


    def apply_flow_gpu_batch(self, f, flow):
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            f_gpu = cp.array(f[ids])
            flow_gpu = cp.array(flow[ids])            
            # Radon transform
            res_gpu = self.apply_flow_gpu(f_gpu,flow_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()            
        return res
  
    def cg_deform_gpu(self, data, psi, flow, titer, xi1=0, rho=0, nproc=16, dbg=False):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi, Tpsi):
            f = cp.linalg.norm(Tpsi-data)**2+rho*cp.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Tpsi = self.apply_flow_gpu(psi, flow)
            grad = (self.apply_flow_gpu(Tpsi-data, -flow) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (cp.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_flow_gpu(d, flow)
            gamma = 0.5*self.line_search(minf, 1, psi,Tpsi,d,Td)
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            if (dbg):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi

    def cg_deform_gpu_batch(self, data, psi, flow, titer, xi1=0, rho=0, nproc=16, dbg=False):
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            data_gpu = cp.array(data[ids])
            psi_gpu = cp.array(psi[ids])    
            xi1_gpu = cp.array(xi1[ids])            
            flow_gpu = cp.array(flow[ids])
            # Radon transform
            res_gpu = self.cg_deform_gpu(data_gpu,psi_gpu,flow_gpu,titer,xi1_gpu,rho,dbg)
            # copy result to cpu
            res[ids] = res_gpu.get()
            
        return res
        



    #SHIFTS




    def apply_shift(self, psi, p):
        """Apply shift for all projections."""
        tmp = cp.zeros([psi.shape[0],2*self.nz, 2*self.n], dtype='float32')
        tmp[:,self.nz//2:3*self.nz//2, self.n//2:3*self.n//2] = psi
        [x,y] = cp.meshgrid(cp.fft.rfftfreq(2*self.n),cp.fft.fftfreq(2*self.nz))
        shift = cp.exp(-2*cp.pi*1j*(x*p[:,1,None,None]+y*p[:,0,None,None]))
        res0 = cp.fft.irfft2(shift*cp.fft.rfft2(tmp))
        res = res0[:,self.nz//2:3*self.nz//2, self.n//2:3*self.n//2]
        return res
    
    def apply_shift_batch(self, u):
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            # Radon transform
            res_gpu = self.apply_shift(u_gpu)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res


    def _upsampled_dft(self, data, ups,
                   upsample_factor=1, axis_offsets=None):
   
        im2pi = 1j * 2 * np.pi
        tdata = data.copy()
        kernel = (cp.tile(cp.arange(ups),(data.shape[0],1))-axis_offsets[:,1:2])[:,:,None]*cp.fft.fftfreq(data.shape[2], upsample_factor)
        kernel = cp.exp(-im2pi * kernel)
        tdata = cp.einsum('ijk,ipk->ijp',kernel,tdata)
        kernel = (cp.tile(cp.arange(ups),(data.shape[0],1))-axis_offsets[:,0:1])[:,:,None]*cp.fft.fftfreq(data.shape[1], upsample_factor)
        kernel = cp.exp(-im2pi * kernel)
        rec = cp.einsum('ijk,ipk->ijp',kernel,tdata)
        
        
        return rec

    def registration_shift(self, src_image, target_image, upsample_factor=1, space="real"):
        
        # assume complex data is already in Fourier space
        if space.lower() == 'fourier':
            src_freq = src_image
            target_freq = target_image
        # real data needs to be fft'd.
        elif space.lower() == 'real':
            src_freq = cp.fft.fft2(src_image)
            target_freq = cp.fft.fft2(target_image)
        
        # Whole-pixel shift - Compute cross-correlation by an IFFT
        shape = src_freq.shape
        image_product = src_freq * target_freq.conj()
        cross_correlation = cp.fft.ifft2(image_product)
        A = cp.abs(cross_correlation)                          
        maxima = A.reshape(A.shape[0],-1).argmax(1)
        maxima = cp.column_stack(cp.unravel_index(maxima,A[0,:,:].shape))

        midpoints = np.array([cp.fix(axis_size / 2) for axis_size in shape[1:]])

        shifts = cp.array(maxima, dtype=cp.float64)
        ids = cp.where(shifts[:,0] > midpoints[0])
        shifts[ids[0],0] -= shape[1]
        ids = cp.where(shifts[:,1] > midpoints[1])
        shifts[ids[0],1] -= shape[2]
        if upsample_factor > 1:
            # Initial shift estimate in upsampled grid
            shifts = np.round(shifts * upsample_factor) / upsample_factor
            upsampled_region_size = np.ceil(upsample_factor * 1.5)
            # Center of output array at dftshift + 1
            dftshift = np.fix(upsampled_region_size / 2.0)
            
            normalization = (src_freq[0].size * upsample_factor ** 2)
            # Matrix multiply DFT around the current shift estimate
        
            sample_region_offset = dftshift - shifts*upsample_factor
            cross_correlation = self._upsampled_dft(image_product.conj(),
                                            upsampled_region_size,
                                            upsample_factor,
                                            sample_region_offset).conj()
            cross_correlation /= normalization
            # Locate maximum and map back to original pixel grid
            A = cp.abs(cross_correlation)                          
            maxima = A.reshape(A.shape[0],-1).argmax(1)
            maxima = cp.column_stack(cp.unravel_index(maxima,A[0,:,:].shape))

            maxima = cp.array(maxima, dtype=cp.float64) - dftshift

            shifts = shifts + maxima / upsample_factor       

        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        for dim in range(src_freq.ndim):
            if shape[dim] == 1:
                shifts[dim] = 0

        
        return shifts
    
    def registration_shift_batch(self, u, w, upsample_factor=1, space="real"):
        res = np.zeros([self.ntheta, 2], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            u_gpu = cp.array(u[ids])
            w_gpu = cp.array(w[ids])
            # Radon transform
            res_gpu = self.registration_shift(u_gpu, w_gpu, upsample_factor, space)
            # copy result to cpu
            res[ids] = res_gpu.get()
        return res

    def cg_shift(self, data, psi, flow, titer, xi1=0, rho=0, dbg=False):
        """CG solver for shift"""
        # minimization functional
        def minf(psi, Tpsi):
            f = np.linalg.norm(Tpsi-data)**2+rho*np.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Tpsi = self.apply_shift_batch(psi, flow)
            grad = (self.apply_shift_batch(Tpsi-data, -flow) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_shift_batch(d, flow)
            gamma = 0.5*self.line_search(minf, 1, psi,Tpsi,d,Td)
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            if (dbg and np.mod(i, 4) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi    

    



    def cg_shift_gpu(self, data, psi, flow, titer, xi1=0, rho=0, dbg=False):
        """CG solver for shift"""
        # minimization functional
        def minf(psi, Tpsi):
            f = cp.linalg.norm(Tpsi-data)**2+rho*cp.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Tpsi = self.apply_shift(psi, flow)            
            #flow = self.registration_shift(data, psi, 1)
            # Tpsi = self.apply_shift(psi, flow)
            # print('a',np.linalg.norm(Tpsi-data))
            
            grad = (self.apply_shift(Tpsi-data, -flow) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+cp.linalg.norm(grad)**2 / \
                    (np.sum(cp.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_shift(d, flow)
            gamma = 0.5*self.line_search(minf, 1, psi,Tpsi,d,Td)
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            if (dbg):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi,flow


    def cg_shift_batch(self, data, psi, flow, titer, xi1=0, rho=0, dbg=False):
        res = np.zeros([self.ntheta, self.nz, self.n], dtype='float32')
        for k in range(0, self.ntheta//self.ptheta):
            ids = np.arange(k*self.ptheta, (k+1)*self.ptheta)
            # copy data part to gpu
            data_gpu = cp.array(data[ids])
            psi_gpu = cp.array(psi[ids])    
            xi1_gpu = cp.array(xi1[ids])            
            flow_gpu = cp.array(flow[ids])
            # Radon transform
            res_gpu,flow_gpu = self.cg_shift_gpu(data_gpu,psi_gpu,flow_gpu,titer,xi1_gpu,rho,dbg)
            # copy result to cpu
            res[ids] = res_gpu.get()
            flow[ids] = flow_gpu.get()
        return res,flow
