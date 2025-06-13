import numpy as np
from scipy.sparse import csr_matrix
from scipy import signal
import networkx as nx
from math import ceil
from numpy.random import random, randn
from tools.createExpansionGraphVARPRO_fast import createExpansionGraphVARPRO_fast
from tools.findLocalMinima import findLocalMinima
import scipy.io
import time
from igraph import Graph


def graphCutIterations(imDataParams,algoParams,residual,lmap, cur_ind):

    DEBUG = 0
    SMOOTH_NOSIGNAL = 1  
    STARTBIG = 1
    dkg = 15 
    DISPLAY_ITER = 0
    gyro = 42.58
    deltaF = np.insert(gyro * np.array(algoParams['species'][1]['frequency']) * imDataParams['FieldStrength'],0,0.0)
    lambda_val = algoParams['lambda']
    dt = imDataParams['TE'][1] - imDataParams['TE'][0]
    period = 1 / dt
    sx, sy, N, C, num_acqs, phase = imDataParams['images'].shape
    fms = np.linspace(algoParams['range_fm'][0], algoParams['range_fm'][1], algoParams['NUM_FMS'])
    dfm = fms[1] - fms[0]
    resoffset = np.arange(0, sx * sy) * algoParams['NUM_FMS']
    masksignal, resLocalMinima, numMinimaPerVoxel = findLocalMinima(residual, 0.06)
    numMinimaPerVoxel = np.squeeze(numMinimaPerVoxel)
    numLocalMin = resLocalMinima.shape[0]
    stepoffset = np.arange(0, sx * sy) * numLocalMin
    stepoffset = stepoffset.flatten(order='F')
    ercurrent = 1e10
    fm = np.zeros((sx, sy))
    fmiters = np.zeros((sx, sy,algoParams['NUM_ITERS']))

    for kg in range(1, algoParams['NUM_ITERS'] + 1):
        fmiters[:,:,kg - 1] = fm
        if kg == 1 and STARTBIG == 1:
            lambdamap = lambda_val * lmap
            ercurrent = 1e10
            prob_bigJump = 1
        elif (kg == dkg and SMOOTH_NOSIGNAL == 1) or STARTBIG == 0:
            lambdamap = lambda_val * lmap
            ercurrent = 1e10
            prob_bigJump = 0.5
        cur_sign = (-1)**kg
        if random() < prob_bigJump:
            cur_ind2 = np.expand_dims(cur_ind, 0)
            repCurInd = np.repeat(cur_ind2, numLocalMin, axis=0)
            repCurInd = np.squeeze(repCurInd)
            if repCurInd.ndim == 2:
                repCurInd = repCurInd[np.newaxis, ...]
            if cur_sign > 0:
                stepLocator = np.logical_and((repCurInd[:,:,:] + 20 / dfm >= resLocalMinima), resLocalMinima > 0)
                stepLocator = np.sum(stepLocator, axis=0) + 1
                validStep = np.logical_and(masksignal > 0, stepLocator <= numMinimaPerVoxel)
            else:
                #data = {'repCurInd': repCurInd, 'resLocalMinima':resLocalMinima}
                #scipy.io.savemat('graphCutIterations.mat', data)

                stepLocator = np.logical_and((repCurInd[:,:,:] - 20 / dfm > resLocalMinima), resLocalMinima > 0)
                stepLocator = np.sum(stepLocator, axis=0)
                validStep = np.logical_and(masksignal > 0, stepLocator >= 1)
            nextValue = np.zeros((sx, sy))
            resLocalMinima1D = resLocalMinima.flatten(order='F')
            stepLocator1D = stepLocator.flatten(order='F')
            nextValue1D = nextValue.flatten(order='F')
            next_ind = stepoffset[validStep.flatten(order='F')] + stepLocator1D[validStep.flatten(order='F')] - 1
            next_ind = next_ind.flatten(order='F')
            nextValue1D[validStep.flatten(order='F')] = resLocalMinima1D[next_ind]
            nextValue = np.reshape(nextValue1D, (sx,sy), order='F')
            cur_step = np.zeros((sx, sy))
            cur_step1d = cur_step.flatten(order='F')
            cur_ind1d = cur_ind.flatten(order='F')
            nextValue1D = nextValue.flatten(order='F')
            cur_step1d[validStep.flatten(order='F')] = nextValue1D[validStep.flatten(order='F')] - cur_ind1d[validStep.flatten(order='F')]
            cur_step = np.reshape(cur_step1d, (sx, sy), order='F')
            if random() < 0.5:
                nosignal_jump = cur_sign * round(abs(deltaF[1]) / dfm)
            else:
                nosignal_jump = cur_sign * abs(round((period - abs(deltaF[1])) / dfm))
            cur_step[~validStep] = nosignal_jump

        else:
            all_jump = cur_sign * ceil(abs(randn() * 3))
            cur_step = all_jump * np.ones((sx, sy))
            nextValue = cur_ind + cur_step
            if cur_sign > 0:
                cur_step1d = cur_step.flatten(order='F')
                cur_ind1d = cur_ind.flatten(order='F')
                cur_step1d[nextValue.flatten(order='F') > len(fms)] = len(fms) - cur_ind1d[nextValue.flatten(order='F') > len(fms)]
                cur_step = np.reshape(cur_step1d, (sx, sy), order='F')
            else:
                cur_step1d = cur_step.flatten(order='F')
                cur_ind1d = cur_ind.flatten(order='F')
                cur_step1d[nextValue.flatten(order='F') < 1] = 1 - cur_ind1d[nextValue.flatten(order='F') < 1]
                cur_step = np.reshape(cur_step1d, (sx, sy), order='F')
        #data = {'cur_step_p': cur_step, 'cur_ind_p':cur_ind}
        #scipy.io.savemat('graphCutIterations.mat', data)

        if np.linalg.norm(lambdamap, ord='fro') > 0:
            A = createExpansionGraphVARPRO_fast(residual, dfm, lambdamap, algoParams['size_clique'], cur_ind, cur_step)
        else:
            A = createExpansionGraphVARPRO_fast(residual, dfm, lambdamap, algoParams['size_clique'], cur_ind, cur_step)
        A[A < 0] = 0
        #gr = digraph(A)
        #flowvalTS, cut_TS = maximum_flow(gr, len(A) - 1, 0)
        
        #G = nx.from_scipy_sparse_matrix(A)
        #cut = nx.minimum_cut(G, A.shape[0]-1, 0,capacity='weight')
        # Convert the Scipy sparse matrix to a dense Numpy array, then to a list of lists
        #start = time.time()
        #matrix_list = A.toarray().tolist()
        #end = time.time()
        #print('toarray: %f'%(end - start))
        #start = time.time()
        g = Graph.Weighted_Adjacency(A, mode="DIRECTED", attr="weight", loops=False)
        #end = time.time()
        #print('Weighted_Adjacency: %f'%(end - start))
        
        min_cut = g.mincut(source=0, target=A.shape[0]-1, capacity='weight')

        #print(cut)
        #print(f"Minimum cut value: {min_cut.value}")
        #print(f"Partition: {min_cut.partition}")

        #cut_orig = cut
        #cut_ind = np.array(list(cut[1][0]), dtype=np.int16)
        cut_ind = np.array((min_cut[1]))

        #print(cut_ind)

        cut1 = np.ones(A.shape[0])
        cut1[cut_ind] = 0
        cut1b = np.ones_like(cut1)
        cut1b[-1] = 0
        data = {'cut1_p': cut1, 'cut1b_p':cut1b}

        if np.sum(A[cut1b == 1, :][:, cut1b == 0]) <= np.sum(A[cut1 == 1, :][:, cut1 == 0]):
            cur_indST = cur_ind
        else:
            cut = np.reshape(cut1[1:-1] == 0, (sx, sy), order='F')
            cur_indST = np.reshape(cur_ind, (sx,sy), order='F') + cur_step * cut
            #erST = np.sum(residual[cur_indST.flatten(order='F') + resoffset]) + dfm**2 * lambdamap[0, 0] * (np.sum(np.abs(np.diff(cur_indST, axis=0))**2) + np.sum(np.abs(np.diff(cur_indST, axis=1))**2))
        prev_ind = cur_ind
        cur_ind = cur_indST
        cur_ind[cur_ind < 1] = 1
        cur_ind[cur_ind > len(fms)] = len(fms)
        #data = {'cut1_p': cut1, 'cut1b_p':cut1b, 'cur_ind_p':cur_ind, 'fms_p':fms}
        #scipy.io.savemat('graphCutIterations.mat', data)
        fm = fms[cur_ind.astype(int)]

    return fm, masksignal

