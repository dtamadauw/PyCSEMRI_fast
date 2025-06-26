import numpy as np

def computeFF( outParams, noise_bias_correction = 1):


            
    curf = outParams['fat_amp']
    curw = outParams['water_amp']


    denom = np.abs(curf) + np.abs(curw)
    denom2 = np.copy(denom)
    denom2[denom==0] = 1  # To avoid divide-by-zero issues
    ff = 100 * np.abs(curf) / denom2

    if noise_bias_correction > 0:
        fatregions = ff > 50
        watregions = ff <= 50
        denom2 = np.abs(curf + curw)
        denom2[denom==0] = 1  # To avoid divide-by-zero issues
        ff[watregions] = 100 - 100 * np.abs(curw[watregions]) / denom2[watregions]
        ff[fatregions] = 100 * np.abs(curf[fatregions]) / denom2[fatregions]

    ff[ff > 125] = 100.0
    ff[ff < -25] = 0.0
    ff[denom == 0.0] = 0.0

    return ff
