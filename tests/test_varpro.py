import numpy as np
import pytest
from pycsemri.VARPRO_LUT import VARPRO_LUT

def generate_synthetic_data(nx=16, ny=16, fm_true=50.0, r2star_true=20.0):
    """Generate synthetic MRI signal for testing."""
    TEs = np.array([1.2, 2.4, 3.6, 4.8, 6.0, 7.2]) * 1e-3
    nTE = len(TEs)
    FieldStrength = 3.0
    GYRO = 42.58
    
    # Fat species (mini model)
    fat_freqs = np.array([-3.8, -3.4, -2.6, -1.9, -0.5, 0.5, 0.6]) * GYRO * FieldStrength
    fat_amps = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.014, 0.035])
    
    images = np.zeros((nx, ny, nTE), dtype=complex)
    
    # Simple model: S = (W + F*b_fat) * exp(i*2pi*fm*t) * exp(-r2s*t)
    for t_idx, t in enumerate(TEs):
        b_fat = np.sum(fat_amps * np.exp(1j * 2 * np.pi * fat_freqs * t))
        signal = (100.0 + 50.0 * b_fat) * np.exp(1j * 2 * np.pi * fm_true * t) * np.exp(-r2star_true * t)
        images[:, :, t_idx] = signal
        
    return images, TEs, FieldStrength

def test_varpro_lut_basic():
    """Test VARPRO_LUT with synthetic data and verify recovery."""
    nx, ny = 16, 16
    fm_true = 50.0
    images, TEs, FieldStrength = generate_synthetic_data(nx, ny, fm_true=fm_true)
    
    imDataParams = {
        'TE': TEs,
        'FieldStrength': FieldStrength,
        'PrecessionIsClockwise': 1,
        'images': images
    }
    
    algoParams = {
        'SUBSAMPLE': 1,
        'mask_threshold': 0.1,
        'range_fm': [-200, 200],
        'NUM_FMS': 41,
        'range_r2star': [0, 100],
        'NUM_R2STARS': 11,
        'species': [
            {'relAmps': [1.0], 'frequency': [0.0]}, # Water
            {
                'relAmps': [0.087, 0.693, 0.128, 0.004, 0.039, 0.014, 0.035],
                'frequency': [-3.8, -3.4, -2.6, -1.9, -0.5, 0.5, 0.6]
            } # Fat
        ]
    }
    
    results = VARPRO_LUT(imDataParams, algoParams)
    
    # Check shape
    assert results['water_amp'].shape == (nx, ny)
    assert results['fm'].shape == (nx, ny)
    
    # Verify field map recovery
    fm_est = results['fm'][nx//2, ny//2]
    assert pytest.approx(fm_est, abs=5.0) == fm_true

if __name__ == "__main__":
    pytest.main([__file__])
