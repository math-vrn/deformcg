import dxchange
import numpy as np
import deformcg as df
import elasticdeform


def deform(data):
    res = data.copy()
    points = [3, 3]
    displacement = (np.random.rand(2, *points) - 0.5) * 10
    for k in range(0, ntheta):
        res[k] = elasticdeform.deform_grid(
            data[k], displacement, order=5, mode='mirror', crop=None, prefilter=True, axis=None)
    return res


if __name__ == "__main__":

    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 2  # number of angles (rotations)
    ptheta = 2  # number of anlge for simultaneous processing on GPU
    # Load object

    u0 = dxchange.read_tiff(
        'data/delta-chip-128.tiff')[:, 64:64+ntheta].swapaxes(0, 1)

    mmin = np.min(u0, axis=(1, 2))
    mmax = np.max(u0, axis=(1, 2))

    with df.SolverDeform(ntheta, nz, n, ptheta) as slv:
        u = deform(u0)
        flow = slv.registration_flow_batch(u0, u, mmin, mmax)
        u1 = slv.apply_flow_gpu_batch(u0, flow)
        u2 = slv.apply_flow_gpu_batch(u1, -flow)

        print('Adjoint test optical flow: ', np.sum(u1*np.conj(u1)),
              '=?', np.sum(u0*np.conj(u2)))

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
        print('Inverse shift error: ', np.linalg.norm(u0-u2)/np.linalg.norm(u0))
