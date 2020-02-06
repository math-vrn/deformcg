import numpy as np
import concurrent.futures as cf
import threading
from scipy import ndimage
from itertools import repeat
from functools import partial
from deformcg.deform import deform
from skimage.feature import register_translation
import cv2

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

    def __init__(self, ntheta, nz, n):
        """Please see help(SolverTomo) for more info."""
        # create class for the tomo transform associated with first gpu
        super().__init__(ntheta, nz, n)

    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        self.free()

    def apply_flow(self, res, f, flow, id):
        """Apply optical flow for one projection."""
        flow0 = flow[id].copy()
        h, w = flow0.shape[:2]
        flow0 = -flow0
        flow0[:, :, 0] += np.arange(w)
        flow0[:, :, 1] += np.arange(h)[:, np.newaxis]
        res[id].real = cv2.remap(f[id].real, flow0,
                                None, cv2.INTER_LANCZOS4)
        #res[id].imag = cv2.remap(f[id].imag, flow0,
         #                       None, cv2.INTER_LANCZOS4)                                 
        return res[id]

    def apply_flow_batch(self, psi, flow):
        """Apply optical flow for all projection in parallel."""
        res = np.zeros(psi.shape, dtype='complex64')
        with cf.ThreadPoolExecutor(32) as e:
            shift = 0
            for res0 in e.map(partial(self.apply_flow, res, psi, flow), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def registration_flow(self, res, psi, g, flow, pars, id):
        """Find optical flow for one projection"""
        tmp1 = psi[id].real  # use only real part
        tmp1 = np.uint8((tmp1-np.min(tmp1)) /
                        (np.max(tmp1)-np.min(tmp1))*255)
        tmp2 = g[id].real
        tmp2 = np.uint8((tmp2-np.min(tmp2)) /
                        (np.max(tmp2)-np.min(tmp2))*255)
        res[id] = cv2.calcOpticalFlowFarneback(
            tmp1, tmp2, flow[id], *pars)
          
        return res[id]

    def registration_flow_batch(self, psi, g, flow=None, pars=[0.5, 3, 20, 16, 5, 1.1, 4]):
        """Find optical flow for all projections in parallel"""
        if (flow is None):
            flow = np.zeros([self.ntheta, self.nz, self.n, 2], dtype='float32')
        res = np.zeros([*psi.shape, 2], dtype='float32')
        with cf.ThreadPoolExecutor(32) as e:
            shift = 0
            for res0 in e.map(partial(self.registration_flow, res, psi, g, flow, pars), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res









    # SHIFT
    def registration_shift(self, res, psi, g, upsample_factor, id):
        """Find x,z shifts for one projection"""
        res[id],_,_ = register_translation(
            psi[id].real, g[id].real, upsample_factor=upsample_factor, space='real')#, return_error=False)
        return res[id]

    def registration_shift_batch(self, psi, g, upsample_factor):
        """Find x,z shifts for all projections in parallel"""
        res = np.zeros([psi.shape[0], 2], dtype='float32')
        with cf.ThreadPoolExecutor(32) as e:
            shift = 0
            for res0 in e.map(partial(self.registration_shift, res, psi, g, upsample_factor), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def apply_shift(self, res, psi, p, id):
        """Apply shift for one projection."""
        # padding to avoid signal wrapping
        tmp = np.zeros([2*self.nz, 2*self.n], dtype='complex64')
        tmp[self.nz//2:3*self.nz//2, self.n//2:3*self.n//2] = psi[id]
        res0 = np.fft.ifft2(
            ndimage.fourier_shift(np.fft.fft2(tmp), p[id]))
        res[id] = res0[self.nz//2:3*self.nz//2, self.n//2:3*self.n//2]
        return res[id]

    def apply_shift_batch(self, psi, flow):
        """Apply shift for all projections in parallel"""
        res = np.zeros(psi.shape, dtype='complex64')
        with cf.ThreadPoolExecutor(32) as e:
            shift = 0
            for res0 in e.map(partial(self.apply_shift, res, psi, flow), range(0, psi.shape[0])):
                res[shift] = res0
                shift += 1
        return res

    def line_search(self, minf, gamma, psi, Tpsi, d, Td):
        """Line search for the step sizes gamma"""
        while(minf(psi, Tpsi)-minf(psi+gamma*d, Tpsi+gamma*Td) < 0):
            gamma *= 0.5
        return gamma

    def cg_deform(self, data, psi, flow, titer, xi1=0, rho=0, dbg=False):
        """CG solver for deformation"""
        # minimization functional
        def minf(psi, Tpsi):
            f = np.linalg.norm(Tpsi-data)**2+rho*np.linalg.norm(psi-xi1)**2
            return f

        for i in range(titer):
            Tpsi = self.apply_flow_batch(psi, flow)
            grad = (self.apply_flow_batch(Tpsi-data, -flow) +
                    rho*(psi-xi1))/max(rho, 1)
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Td = self.apply_flow_batch(d, flow)
            gamma = 0.5*self.line_search(minf, 1, psi,Tpsi,d,Td)
            grad0 = grad
            # update step
            psi = psi + gamma*d
            # check convergence
            if (dbg and np.mod(i, 1) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi

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
            if (dbg and np.mod(i, 1) == 0):
                print("%4d, %.3e, %.7e" %
                      (i, gamma, minf(psi, Tpsi+gamma*Td)))
        return psi    


