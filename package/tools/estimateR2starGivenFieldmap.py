import numpy as np
from tools.getPhiMatrixMultipeak import getPhiMatrixMultipeak


def estimateR2starGivenFieldmap(imDataParams, algoParams, fm):

    try:
        precessionIsClockwise = imDataParams['PrecessionIsClockwise']
    except KeyError:
        precessionIsClockwise = 1

    if precessionIsClockwise <= 0:
        imDataParams['images'] = np.conj(imDataParams['images'])
        imDataParams['PrecessionIsClockwise'] = 1

    range_r2star = algoParams['range_r2star']
    NUM_R2STARS = round(algoParams['range_r2star'][1]/2)+1; 
    gyro = 42.58
    deltaF = np.concatenate(([0], gyro * (np.array(algoParams['species'][1]['frequency']) - algoParams['species'][0]['frequency'][0]) * imDataParams['FieldStrength']))
    relAmps = algoParams['species'][1]['relAmps']
    images = imDataParams['images']
    t = imDataParams['TE']
    t = np.reshape(t, (-1, 1))

    sx, sy, _, C, N, _ = images.shape
    num_acqs = 1  # Assuming there's only one acquisition

    images = np.transpose(images, (0, 1, 4, 3, 5, 2))
    images = np.reshape(images, (sx, sy, N, C * num_acqs))

    images2 = np.zeros_like(images, dtype=complex)
    for kt in range(N):
        for kc in range(C * num_acqs):
            images2[:, :, kt, kc] = images[:, :, kt, kc] * np.exp(-1j * 2 * np.pi * fm * t[kt])


    r2s = np.linspace(range_r2star[0], range_r2star[1], NUM_R2STARS)
    Phi = getPhiMatrixMultipeak(deltaF, relAmps, t)  # Assuming this function is defined elsewhere
    P = np.empty((0,6))
    for k in range(NUM_R2STARS):
        Psi = np.diag(np.exp(-r2s[k] * np.squeeze(t)))
        P = np.concatenate((P,np.eye(N) - np.matmul(np.matmul(Psi, Phi,), np.linalg.pinv(np.matmul(Psi, Phi))  )), axis=0)

        
    residual = np.zeros((sx, sy, NUM_R2STARS))

    for kx in range(sx):
        temp = np.transpose(np.squeeze(images2[kx, :, :, :]), (1, 0))
        temp = np.reshape(temp, (N, sy * C * num_acqs))
        temp2 = np.reshape(np.sum(np.abs( np.reshape(np.matmul(P, temp), (N, C*num_acqs*NUM_R2STARS*sy)))** 2, axis=0), (NUM_R2STARS, C * num_acqs * sy))
        temp2 = np.transpose(temp2)
        temp2 = np.sum(np.reshape(temp2, (C * num_acqs, NUM_R2STARS * sy)), axis=0)
        residual[kx, :, :] = np.reshape(temp2, (sy, NUM_R2STARS))

    minres, iminres = np.min(residual, axis=2), np.argmin(residual, axis=2)
    r2starmap = r2s[iminres]

    return r2starmap, residual