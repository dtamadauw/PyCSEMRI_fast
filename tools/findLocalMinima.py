import numpy as np
import scipy.io


def findLocalMinima(residual, threshold, masksignal=None):
    if residual.ndim > 3:
        L = residual.shape[0]
        sx = residual.shape[1]
        sy = residual.shape[2]
        sz = residual.shape[3]
    else:
        L = residual.shape[0]
        sx = residual.shape[1]
        sy = residual.shape[2]
        sz = 1
    dres = np.diff(residual, axis=0)
    maxres = np.squeeze(np.max(residual, axis=0))
    minres = np.squeeze(np.min(residual, axis=0))
    if masksignal is None:
        sumres = np.sqrt(np.squeeze(np.sum(residual, axis=0)))
        sumres = sumres / np.max(sumres)
        masksignal = sumres > threshold
    resLocalMinima = np.zeros((1, sx, sy))
    alloc_resLocalMinima = False
    numMinimaPerVoxel = np.zeros((sx, sy, 1))



    for kx in range(sx):
        for ky in range(sy):
            if masksignal[kx, ky] > 0:
                minres = np.min(residual[:, kx, ky])
                maxres = np.max(residual[:, kx, ky])
                temp = np.concatenate(([0], np.squeeze(dres[:, kx, ky])))
                temp = np.logical_and(np.logical_and(temp < 0, np.roll(temp, -1) > 0), residual[:, kx, ky] < minres + 0.3 * (maxres - minres))
                if not alloc_resLocalMinima:
                    resLocalMinima = np.zeros(((np.sum(temp)), sx, sy))
                    alloc_resLocalMinima = True
                if np.sum(temp) > resLocalMinima.shape[0]:
                    resLocalMinima = np.pad(resLocalMinima, ((np.sum(temp)-resLocalMinima.shape[0],0),(0,0),(0,0)))

                #data = {'temp': temp}
                #scipy.io.savemat('findLocalMinima.mat', data)
 
                #res_temp = np.where(temp)[0][0]
                #resLocalMinima[0:(np.sum(temp)), kx:(kx+1), ky:(ky+1)] = np.where(temp)[0][0].reshape(-1, 1, 1)
                resLocalMinima[0:(np.count_nonzero(temp)), kx:(kx+1), ky:(ky+1)] = np.where(temp != 0)[0].reshape(-1, 1, 1)
                numMinimaPerVoxel[kx, ky] = np.sum(temp)
    return masksignal, resLocalMinima, numMinimaPerVoxel

'''
def findLocalMinima(residual, threshold, masksignal=None):
    if residual.ndim > 3:
        L = residual.shape[0]
        sx = residual.shape[1]
        sy = residual.shape[2]
        sz = residual.shape[3]
    else:
        L = residual.shape[0]
        sx = residual.shape[1]
        sy = residual.shape[2]
        sz = 1
    dres = np.diff(residual, axis=0)
    maxres = np.squeeze(np.max(residual, axis=0))
    minres = np.squeeze(np.min(residual, axis=0))
    if masksignal is None:
        sumres = np.sqrt(np.squeeze(np.sum(residual, axis=0)))
        sumres = sumres / np.max(sumres)
        masksignal = sumres > threshold
    resLocalMinima = np.zeros((1, sx, sy))
    alloc_resLocalMinima = False
    numMinimaPerVoxel = np.zeros((sx, sy, 1))



    for kx in range(sx):
        for ky in range(sy):
            if masksignal[kx, ky] > 0:
                minres = np.min(residual[:, kx, ky])
                maxres = np.max(residual[:, kx, ky])
                temp = np.concatenate(([0], np.squeeze(dres[:, kx, ky])))
                temp = np.logical_and(np.logical_and(temp < 0, np.roll(temp, -1) > 0), residual[:, kx, ky] < minres + 0.3 * (maxres - minres))
                if not alloc_resLocalMinima:
                    resLocalMinima = np.zeros(((np.sum(temp)), sx, sy))
                    alloc_resLocalMinima = True
                #res_temp = np.where(temp)[0][0]
                #resLocalMinima[0:(np.sum(temp)), kx:(kx+1), ky:(ky+1)] = np.where(temp)[0][0].reshape(-1, 1, 1)
                resLocalMinima[0:(np.sum(temp)), kx:(kx+1), ky:(ky+1)] = np.where(temp != 0)[0].reshape(-1, 1, 1)
                numMinimaPerVoxel[kx, ky] = np.sum(temp)
    return masksignal, resLocalMinima, numMinimaPerVoxel
'''


