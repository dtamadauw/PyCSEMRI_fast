import numpy as np

def getPhiMatrixMultipeak(deltafs, relAmps, t):
    DF, T = np.meshgrid(deltafs, t)
    A, T2 = np.meshgrid(relAmps, t)

    Phi1 = np.exp(1j * 2 * np.pi * T * DF)
    Phi = np.column_stack((Phi1[:, 0], np.sum(Phi1[:, 1:] * A, axis=1)))

    return Phi
