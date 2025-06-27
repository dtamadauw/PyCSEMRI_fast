import numpy as np
from scipy.sparse import csr_matrix
from scipy import signal
from math import ceil
from numpy.random import random, randn
from tools.createExpansionGraphVARPRO_fast import createExpansionGraphVARPRO_fast
from tools.getQuadraticApprox import getQuadraticApprox
from tools.computeResidual import computeResidual
from tools.estimateR2starGivenFieldmap import estimateR2starGivenFieldmap
from decomposeGivenFieldMapAndDampings import decomposeGivenFieldMapAndDampings
from graphCutIterations import graphCutIterations
import scipy.io
from scipy.interpolate import griddata
import os # Import the os module
import time


def fw_i2cm1i_graphcut(imDataParams_in, algoParams_in):
    imDataParams = imDataParams_in.copy()
    algoParams = algoParams_in.copy()
    
    images_orig = imDataParams['images']
    PrecessionIsClockwise_orig = imDataParams['PrecessionIsClockwise']
    DEBUG = 0
    SUBSAMPLE = algoParams['SUBSAMPLE']
    if SUBSAMPLE > 1:
        sx = imDataParams['images'].shape[0]
        sy = imDataParams['images'].shape[1]
        images0 = imDataParams['images']
        START = round(SUBSAMPLE / 2)-1
        allX = np.arange(0, sx)
        allY = np.arange(0, sy)
        subX = np.arange(START, sx, SUBSAMPLE)
        subY = np.arange(START, sy, SUBSAMPLE)
        imDataParams['images'] = images0[np.ix_(subX, subY, np.arange(images0.shape[2]), np.arange(images0.shape[3]), np.arange(images0.shape[4]))]
    if 'residual' in algoParams:
        residual = algoParams['residual']
    else:
        dTE = np.diff(imDataParams['TE'])
        TRY_PERIODIC_RESIDUAL = algoParams.get('TRY_PERIODIC_RESIDUAL', 1)
        if TRY_PERIODIC_RESIDUAL == 1 and np.sum(np.abs(dTE - dTE[0])) < 1e-6:
            UNIFORM_TEs = 1
        else:
            UNIFORM_TEs = 0
        if UNIFORM_TEs == 1:
            dt = imDataParams['TE'][1] - imDataParams['TE'][0]
            period = np.abs(1 / dt)
            NUM_FMS_ORIG = algoParams['NUM_FMS']
            range_val = np.diff(algoParams['range_fm'])
            params = algoParams.copy()
            params['NUM_FMS'] = np.ceil(algoParams['NUM_FMS'] / range_val * period)
            params['range_fm'] = [0, period * (1 - 1 / (algoParams['NUM_FMS']))]
            residual = computeResidual(imDataParams, params)
            num_periods = np.ceil(range_val / period / 2)
            algoParams['NUM_FMS'] = 2 * num_periods * algoParams['NUM_FMS']
            residual = np.tile(residual, [2 * num_periods, 1, 1])
            algoParams['range_fm'] = [-num_periods * period, (num_periods * period - period / NUM_FMS_ORIG)]
        else:
            residual = computeResidual(imDataParams, algoParams)
            
    L = residual.shape[0]
    sx = imDataParams['images'].shape[0]
    sy = imDataParams['images'].shape[1]
    sz = imDataParams['images'].shape[2]
    if sz > 1:
        print('Multi-slice data: processing central slice')
        imDataParams['images'] = imDataParams['images'][:, :, np.ceil(sz / 2), :, :]
    C = imDataParams['images'].shape[3]
    if C > 1:
        print('Multi-coil data is not supported in Python version')
    if imDataParams['PrecessionIsClockwise'] <= 0:
        imDataParams['images'] = np.conj(imDataParams['images'])
        imDataParams['PrecessionIsClockwise'] = 1
    SUBSAMPLE = algoParams['SUBSAMPLE']
    lambda_val = algoParams['lambda']
    LMAP_POWER = algoParams['LMAP_POWER']
    LMAP_EXTRA = algoParams['LMAP_EXTRA']
    DO_OT = algoParams['DO_OT']
    fms = np.linspace(algoParams['range_fm'][0], algoParams['range_fm'][1], algoParams['NUM_FMS'])
    dfm = fms[1] - fms[0]
    lmap = np.abs(getQuadraticApprox(residual, dfm))
    lmap = (np.sqrt(lmap)) ** LMAP_POWER
    lmap = lmap + np.mean(lmap) * LMAP_EXTRA
    #data = {'lmap': lmap, 'residual':residual, 'dfm':dfm}
    #scipy.io.savemat('lmap.mat', data)
    cur_ind = np.ceil(len(fms) / 2) * np.ones(imDataParams['images'][:, :, 0, 0, 0].shape)

    fm, masksignal = graphCutIterations(imDataParams, algoParams, residual, lmap, cur_ind)
    #fm, masksignal = graphCutIterations_native(imDataParams, algoParams, residual, lmap, cur_ind)
    if SUBSAMPLE > 1:
        fmlowres = fm
        SUBX, SUBY = np.meshgrid(subY, subX)
        ALLX, ALLY = np.meshgrid(allY, allX)
        points = np.vstack([SUBX.ravel(), SUBY.ravel()]).T
        values = fmlowres.flatten()
        values_mask = masksignal.flatten()
        grid_points = np.vstack([ALLX.ravel(), ALLY.ravel()]).T
        fm = griddata(points, values, grid_points, method='cubic').reshape(ALLX.shape)
        masksignal = griddata(points, values_mask, grid_points, method='cubic').reshape(ALLX.shape)
        fm[np.isnan(fm)] = 0
        # Replace NaN with 0
        masksignal[np.isnan(masksignal)] = 0

        
    imDataParams['images'] = images_orig
    imDataParams['PrecessionIsClockwise'] = PrecessionIsClockwise_orig


    start = time.time()
    r2starmap, residual_r2s = estimateR2starGivenFieldmap(imDataParams, algoParams, fm)
    end = time.time()
    print("Process Time estimateR2starGivenFieldmap: %f" % (end - start))

    start = time.time()
    amps = decomposeGivenFieldMapAndDampings( imDataParams,algoParams,fm,r2starmap,r2starmap )
    end = time.time()
    print("Process Time decomposeGivenFieldMapAndDampings: %f" % (end - start))


    w = np.squeeze(amps[:,:,0,:])
    f = np.squeeze(amps[:,:,1,:])


    return {'fm': fm, 'amp_w': w, 'amp_f': f, 'r2starmap': r2starmap, 'masksignal': masksignal}

