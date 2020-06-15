import dxchange
import numpy as np
import deformcg as df

if __name__ == "__main__":
    # Model parameters
    n = 128  # object size n x,y
    nz = 128  # object size in z
    ntheta = 1  # number of angles (rotations)
    ptheta = 1

    # Load object
    u0 = dxchange.read_tiff(
        'data/delta-chip-128.tiff')[:, 64:64+ntheta].swapaxes(0, 1)

    shifts0 = np.random.random([ntheta, 2]).astype('float32')*10
    with df.SolverDeform(ntheta, nz, n, ptheta) as slv:
        u = slv.apply_shift_batch(u0, shifts0)
        shifts = slv.registration_shift_batch(u, u0, upsample_factor=10)
        print(shifts)
        rec = slv.apply_shift_batch(u, -shifts)
    print('error:', np.linalg.norm(rec-u0)/np.linalg.norm(u0))
    dxchange.write_tiff(
        u0[ntheta//2], 'resshift/delta-chip-128.tiff', overwrite=True)
    dxchange.write_tiff(
        u[ntheta//2], 'resshift/shiftdelta-chip-128.tiff', overwrite=True)
    dxchange.write_tiff(
        rec[ntheta//2], 'resshift/recdefdelta-chip-128.tiff', overwrite=True)
