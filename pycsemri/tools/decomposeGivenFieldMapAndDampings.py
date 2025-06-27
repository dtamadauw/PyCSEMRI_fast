import numpy as np



def decomposeGivenFieldMapAndDampings( imDataParams,algoParams,fieldmap,r2starWater,r2starFat ):

    gyro = 42.58

    try:
        precessionIsClockwise = imDataParams['PrecessionIsClockwise']
    except KeyError:
        precessionIsClockwise = 1

    try:
        ampW = algoParams['species'][0]['relAmps']
    except KeyError:
        ampW = 1.0


    # If precession is clockwise (positive fat frequency) simply conjugate data
    if precessionIsClockwise <= 0:
        images = np.conj(imDataParams['images'])

    deltaF = np.concatenate((np.array([0]), gyro*(algoParams['species'][1]['frequency'] - algoParams['species'][0]['frequency'][0])*(imDataParams['FieldStrength'])))

    relAmps = algoParams['species'][1]['relAmps']
    images = imDataParams['images']
    t = imDataParams['TE']

    sx, sy, _, C, N, _ = images.shape
    relAmps = np.reshape(relAmps, (1, -1))



    B1 = np.zeros((N, 2), dtype=complex)
    B = np.zeros((N, 2), dtype=complex)
    for n in range(N):
        B1[n, 0] = ampW * np.exp(1j * 2 * np.pi * deltaF[0] * t[n])
        B1[n, 1] = np.array(np.sum(relAmps * np.exp(1j * 2 * np.pi * deltaF[1:] * t[n])))

    remerror = np.zeros((sx, sy))
    amps = np.zeros((sx, sy, 2, C), dtype=complex)
    for kx in range(sx):
        for ky in range(sy):
            s = np.squeeze(images[kx, ky, :, :, :])

            B[:, 0] = B1[:, 0] * np.exp(1j * 2 * np.pi * fieldmap[kx, ky] * t - r2starWater[kx, ky] * t)
            B[:, 1] = B1[:, 1] * np.exp(1j * 2 * np.pi * fieldmap[kx, ky] * t - r2starFat[kx, ky] * t)

            amps[kx, ky, :, :] = np.reshape(np.linalg.lstsq(B, s, rcond=None)[0], (2,1))

    return amps