import numpy as np
import scipy.io


def getPhiMatrixMultipeak(deltafs, relAmps, t):
    DF, T = np.meshgrid(deltafs, t)
    A, T2 = np.meshgrid(relAmps, t)
    Phi1 = np.exp(1j * 2 * np.pi * T * DF)
    Phi = np.column_stack((Phi1[:, 0], np.sum(Phi1[:, 1:] * A, axis=1)))
    return Phi


def computeResidual(imDataParams, algoParams):
    images = imDataParams['images']
    precessionIsClockwise = imDataParams.get('PrecessionIsClockwise', 1)
    if precessionIsClockwise <= 0:
        imDataParams['images'] = np.conj(imDataParams['images'])
        imDataParams['PrecessionIsClockwise'] = 1
    gyro = 42.58
    deltaF = np.insert(gyro * (algoParams['species'][1]['frequency'] - algoParams['species'][0]['frequency']) * imDataParams['FieldStrength'], 0, 0.0)
    range_fm = algoParams['range_fm']
    t = imDataParams['TE']
    NUM_FMS = algoParams['NUM_FMS']
    range_r2star = algoParams['range_r2star']
    NUM_R2STARS = algoParams['NUM_R2STARS']
    sx = images.shape[0]
    sy = images.shape[1]
    N = images.shape[4]
    C = images.shape[3]
    num_acq = images.shape[5]
    Phi = getPhiMatrixMultipeak(deltaF, algoParams['species'][1]['relAmps'], t)
    iPhi = np.linalg.pinv(Phi.conj().T @ Phi)
    A = Phi @ iPhi @ Phi.conj().T
    psis = np.linspace(range_fm[0], range_fm[1], NUM_FMS)
    r2s = np.linspace(range_r2star[0], range_r2star[1], NUM_R2STARS)
    P = np.empty((NUM_FMS*t.shape[0], N, 0))
    for kr in range(NUM_R2STARS):
        P1 = np.empty((0,N))
        for k in range(NUM_FMS):
            Psi = np.diag(np.exp(1j * 2 * np.pi * psis[k] * t - np.abs(t) * r2s[kr]))
            P1 = np.append(P1, np.eye(N) - Psi @ Phi @ np.linalg.pinv(Psi @ Phi), axis=0)
        P1 = np.array(P1)
        P = np.append(P, P1[:,:,np.newaxis], axis=2)
    residual = np.zeros((NUM_FMS, sx, sy))
    for ka in range(num_acq):
        for ky in range(sy):
            image_tr = np.transpose(images[:, :, :, :, :, ka], [0, 1, 2, 4, 3])
            temp = np.reshape(np.squeeze(image_tr[:,ky,:,:,:]), [sx, N*C]).conj().T
            temp = np.reshape(temp, [N, sx * C])
            temp3 = np.empty((0,NUM_FMS * sx))
            
            a = np.array(P[:,:,0]) @ temp
            for kr in range(NUM_R2STARS):
                bb = np.abs(np.reshape(np.array(P[:,:,kr]) @ temp, [N, C * NUM_FMS * sx],order='F'))
                temp2 = np.reshape(np.sum(bb ** 2, axis=0), [NUM_FMS, C * sx],order='F').conj().T
                temp3 = np.append(temp3, np.reshape( np.sum(np.reshape(temp2, [C, NUM_FMS * sx],order='F'), axis=0), (1, NUM_FMS * sx) ), axis=0 )

            mint3, imint3 = np.min(temp3, axis=0), np.argmin(temp3, axis=0)

            residual[:, :, ky] = np.squeeze( np.squeeze( residual[:, :, ky]).conj().T + np.reshape(mint3, [sx, NUM_FMS],order='F') ).conj().T    
    
    #data = {'images': images, 'P': P, 'mint3':mint3, 'temp': temp, 'temp2':temp2, 'temp3':temp3, 'residual':residual}
    #scipy.io.savemat('computeResidual.mat', data)

    return residual
