import numpy as np
from scipy.sparse import csr_matrix
import scipy.io



def createExpansionGraphVARPRO_fast(residual, dfm, lambda_val, size_clique, cur_ind, step):
    sx = residual.shape[1]
    sy = residual.shape[2]
    L = residual.shape[0]
    s = sx * sy
    num_nodes = s + 2
    num_edges = num_nodes * (2 + (size_clique + 1) ** 2)
    sA = [num_nodes, num_nodes]
    offset = np.arange(0, s) * L
    cur_ind = np.squeeze(cur_ind)
    step_ind = cur_ind[:,:] + step
    step_ind = np.reshape(step_ind, (sx,sy,1),order='F')
    cur_ind1d = cur_ind.flatten(order='F')
    step_ind1d = step_ind.flatten(order='F')


    if np.issubdtype(type(residual), np.integer):
        maxA = 1e5
    else:
        maxA = 1e6
    valsh = np.zeros((1, s))
    valsv = np.zeros((s, 1))
    factor = lambda_val * dfm ** 2
    factor1d = factor.flatten(order='F')
    x = np.arange(1, sx + 1)
    y = np.arange(1, sy + 1)
    Y, X = np.meshgrid(y, x)
    allIndCross = []
    allValsCross = []
    for dx in range(-size_clique, size_clique + 1):
        for dy in range(-size_clique, size_clique + 1):
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist > 0:
                validmapi = np.logical_and(np.logical_and(X + dx >= 1, X + dx <= sx), np.logical_and(Y + dy >= 1, Y + dy <= sy))
                validmapi = validmapi.flatten(order='F')
                validmapj = np.logical_and(np.logical_and(X - dx >= 1, X - dx <= sx), np.logical_and(Y - dy >= 1, Y - dy <= sy))
                validmapj = validmapj.flatten(order='F')
                curfactor = np.minimum(factor1d[validmapi], factor1d[validmapj])
                #curfactor = np.reshape(curfactor, (len(curfactor),1),order='F')
                curfactor = curfactor.flatten(order='F')
                a = curfactor * (1. / dist * (cur_ind1d[validmapi] - cur_ind1d[validmapj]) ** 2)
                b = curfactor * (1. / dist * (cur_ind1d[validmapi] - step_ind1d[validmapj]) ** 2)
                c = curfactor * (1. / dist * (step_ind1d[validmapi] - cur_ind1d[validmapj]) ** 2)
                d = curfactor * (1. / dist * (step_ind1d[validmapi] - step_ind1d[validmapj]) ** 2)
                temp = np.zeros((1, s))
                temp[0,validmapi.flatten(order='F')] = np.reshape(np.maximum(0, c - a), (1,-1),order='F')
                valsh = valsh + temp
                temp = np.zeros((s, 1))
                temp[validmapi.flatten(order='F'),0] = np.maximum(0, a - c)
                valsv = valsv + temp
                temp = np.zeros((1, s))
                temp[0,validmapj.flatten(order='F')] = np.reshape(np.maximum(0, d - c), (1,-1),order='F')
                valsh = valsh + temp
                temp = np.zeros((s, 1))
                temp[validmapj.flatten(order='F'),0] = np.maximum(0, c - d)
                valsv = valsv + temp
                S = np.arange(1, s + 1)
                #Sh = 1 + (S[validmapi.flatten(order='F')])
                #Sv = 1 + (S[validmapj.flatten(order='F')])
                Sh = (S[validmapi.flatten(order='F')])
                Sv = (S[validmapj.flatten(order='F')])
                indcross = np.ravel_multi_index((Sh, Sv), sA)
                temp = b + c - a - d
                allIndCross = np.append(allIndCross, indcross.flatten(order='F'))
                allValsCross = np.append(allValsCross, temp.flatten(order='F'))

    rows = np.ones(num_nodes - 2, dtype=int)
    cols = np.arange(1, num_nodes-1)
    ind1 = np.ravel_multi_index((rows,cols), dims=sA, order='F') - 1
    rows = np.arange(1, num_nodes-1)
    cols = np.zeros(num_nodes - 2, dtype=int)+num_nodes-1
    ind2 = np.ravel_multi_index((rows,cols), dims=sA, order='F')

    residual1D = residual.flatten(order='F')
    temp0 = residual1D[(cur_ind.flatten(order='F') + offset.flatten(order='F')).astype(int)]
    valid_ind = np.logical_and(step_ind.flatten(order='F') >= 1, step_ind.flatten(order='F') <= L)
    temp1 = np.zeros((s,1))
    step_ind1d = step_ind.flatten(order='F')
    temp1_ind = (step_ind1d[valid_ind].flatten(order='F') + offset[valid_ind]).astype(int) - 1
    #print("%d, %d, %d" %(np.max(step_ind1d[valid_ind]), np.max(offset[valid_ind]), np.max(temp1_ind)) )
    temp1[valid_ind,0] = residual1D[temp1_ind]
    curmaxA = np.maximum(np.max(temp0), np.max(np.concatenate((valsh.flatten(order='F'), valsv.flatten(order='F'), allValsCross))))
    infty = curmaxA
    temp1[~valid_ind] = infty
    indAll = np.concatenate((ind1, ind2, allIndCross))
    valuesAll = np.concatenate((valsh.flatten(order='F') + np.reshape(np.maximum(temp1.flatten(order='F') - temp0, 0), (1, s)).flatten(order='F'), valsv.flatten(order='F') + np.reshape(np.maximum(0, temp0 - temp1.flatten(order='F')), (s, 1)).flatten(order='F'), allValsCross.flatten(order='F')))
    indAllSort = np.sort(indAll).astype(int)
    sortIndex = np.argsort(indAll)
    valsort = valuesAll[sortIndex]
    xind, yind = np.unravel_index(indAllSort, sA, order='F')
    A = csr_matrix((valsort, (xind, yind)), shape=(num_nodes, num_nodes))
    A = np.round(A * maxA / curmaxA)
    A[A < 0] = 0

    # Assuming A is your SciPy sparse array


    #data = {'A_p': A, 'ind1_p': ind1, 'ind2_p':ind2, 'residual1D_p':residual1D, 'valsort_p':valsort, 'allIndCross_p':allIndCross, 'allValsCross_p':allValsCross, 'temp_p':temp, 'a_p':a, 'b_p':b, 'c_p':c, 'd_p':d,\
    #         'valuesAll_p':valuesAll, 'curmaxA_p':curmaxA, 'cur_ind_p':cur_ind, 'step_ind_p':step_ind, 'valid_ind_p':valid_ind, 'temp1_p':temp1, 'step_ind1d_p':step_ind1d, 'valsh_p':valsh, 'valsv_p':valsv, 'step_p':step,\
    #            'xind_p':xind, 'yind_p':yind, 'indAllSort_p':indAllSort, 'sA_p':sA, 'indAll_p':indAll, 'sortIndex_p':sortIndex}
    #scipy.io.savemat('createExpansionGraphVARPRO_fast.mat', data)
    return A
