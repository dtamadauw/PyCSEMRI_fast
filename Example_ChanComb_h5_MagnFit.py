import h5py
import numpy as np
from scipy.io import loadmat
from fw_i2cm1i_graphcut import fw_i2cm1i_graphcut
from fwFit_MixedLS_1r2star import fwFit_MixedLS_1r2star
from computeFF import computeFF
from create_robust_mask import create_robust_mask


# Open the HDF5 file
file = h5py.File('/home/dxt005/home/Tools/WaterFatPython/Data/invivo/IdealChanCombData_16415_2.h5', 'r')

xres = file['Header']['ImageXRes'][0]
yres = file['Header']['ImageYRes'][0]
nslices = file['Header']['NumSlices'][0]
nTE = file['Header']['NumEchoes'][0]
TEs = file['Header']['EchoTimes'][0:nTE]
FieldStrength = file['Header']['FieldStrength'][0]

ims = np.zeros((xres, yres, nTE, nslices), dtype=np.cdouble)

# Iterate over each key
for sl in range(len(file['Data'].keys())):
    # Get the data
    key = 'Slice%d' % (sl)
    data = np.array(list(file['Data'][key]))
    data_c = data['real'] + 1j*data['imag']
    ims[:,:,:,sl] = data_c.transpose((2,1,0))



# Setup params
PROCESS_PHANTOM = 0


fatAmps = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
fatFreq = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
if PROCESS_PHANTOM==1:
	fatFreq = fatFreq - 0.08

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
    'TE': TEs,
    'FieldStrength':FieldStrength/10000.0,
    'PrecessionIsClockwise': -1,
    'images':np.array([0, 0])
}


R2s3D = np.zeros((xres, yres, nslices))
PDFF3D = np.zeros((xres, yres, nslices))

for sl in range(10,26):

    print('Slice %d'%(sl))
    
    imDataParams['images'] = ims[:,:,:,sl].reshape(xres, yres, 1, 1, nTE,1)
      
    print('fw_i2cm1i_graphcut')

    initParams = fw_i2cm1i_graphcut(imDataParams, algoParams)
    mask = create_robust_mask(np.abs(ims[:,:,0,sl]), percentile_low=0, percentile_high=50)
    initParams['masksignal'] = mask

    print('fwFit_MagnLS_1r2star')

    outParams = fwFit_MagnLS_1r2star(imDataParams, algoParams, initParams)

    print('In the loop')
    R2s3D[:,:,sl] = outParams['r2starmap']
    print('Calc PDFF')
    PDFF3D[:,:,sl] = computeFF( outParams )
    print('Next Slice')



