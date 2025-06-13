import numpy as np
from scipy.io import loadmat
from fw_i2cm1i_graphcut import fw_i2cm1i_graphcut
from fwFit_ComplexLS_1r2star import fwFit_ComplexLS_1r2star
import ctypes


# Load the .mat file
img_recon_python = loadmat('image.mat')
img_recon = img_recon_python['img_recon']
img_recon2D = img_recon[:,:,2,:].reshape(144, 144, 1, 1, 6,1)



# Setup params
fatAmps = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
fatFreq = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])

algoParams = {
    'species': [
        {
            'name': 'water',
            'frequency': np.array([0.0]),
            'relAmps': np.array([1.0])
        },
        {
            'name': 'fat',
            'frequency': fatFreq,
            'relAmps': fatAmps
        }
    ],
    'size_clique':1,
    'range_r2star':np.array([0, 1000]),
    'NUM_R2STARS':11,
    'range_fm':np.array([-600, 600]),
    'NUM_FMS':101,
    'NUM_ITERS':40,
    'SUBSAMPLE':4,#4
    'DO_OT':0,
    'LMAP_POWER':2,
    'lambda':0.02,
    'LMAP_EXTRA':0.02,
    'TRY_PERIODIC_RESIDUAL':0
}

imDataParams = {
    'TE': np.array([0.0011, 0.0020, 0.0028, 0.0036, 0.0044, 0.0053]),
    'FieldStrength':3.0,
    'PrecessionIsClockwise': -1,
    'images':np.array([0, 0])
}

imDataParams['images']=img_recon2D


initParams = fw_i2cm1i_graphcut(imDataParams, algoParams)

outParams = fwFit_ComplexLS_1r2star(imDataParams, algoParams, initParams)