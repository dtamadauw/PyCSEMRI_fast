import numpy as np
import scipy.io


def getQuadraticApprox(residual, dfm):
    NUM_FMS, sx, sy = residual.shape
    resoffset = np.arange(0, sx * sy) * NUM_FMS
    minres, iminres = np.min(residual[9:-9, :, :], axis=0), np.argmin(residual[9:-9, :, :], axis=0)
    iminres = np.squeeze(iminres + 9)
    residual1d = residual.flatten(order='F')
    d2 = (residual1d[(iminres.flatten(order='F') + 1) + resoffset] + residual1d[(iminres.flatten(order='F') - 1) + resoffset] - 2 * residual1d[(iminres.flatten(order='F')) + resoffset]) / dfm ** 2
    d2 = np.reshape(d2, [sx, sy],order='F')

    #data = {'iminres': iminres, 'resoffset': resoffset, 'd2':d2}
    #scipy.io.savemat('getQuadraticApprox.mat', data)

    return d2
