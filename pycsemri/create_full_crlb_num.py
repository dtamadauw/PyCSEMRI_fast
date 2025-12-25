import numpy as np

def create_full_crlb_num(rho_W, rho_F, phi, r2, TE1, deltaTE, B0, N, sigma):
    """
    Calculates the Cramér-Rao Lower Bound (CRLB) for multi-point DIXON MRI signals.

    This is a Python translation of the original MATLAB script. It computes the theoretical
    minimum variance for estimates of fat fraction and R2* from a multi-echo gradient
    echo signal, given a specific set of acquisition and tissue parameters.

    Args:
        rho_W (float): Water signal amplitude (real).
        rho_F (float): Fat signal amplitude (real).
        phi (float): General off-resonance phase in radians.
        r2 (float): R2* value (1/T2*).
        TE1 (float): First echo time in seconds.
        deltaTE (float): Echo spacing in seconds.
        B0 (float): Main magnetic field strength in Tesla.
        N (int): Number of echoes.
        sigma (float): Standard deviation of the noise in the measurements.

    Returns:
        tuple[float, float]: A tuple containing:
            - std_dev_F_percent (float): The CRLB for the fat fraction, expressed as a
              standard deviation in percent.
            - var_R2s (float): The CRLB for the R2* estimate (variance).
    """
    # --- 1. Setup Parameters and Constants ---
    
    t = TE1 + np.arange(N) * deltaTE
    a = np.array([[rho_W, rho_F]])
    gamma = 42.58e6
    rel_amps = np.array([0.087, 0.693, 0.128, 0.004, 0.039, 0.048])
    fat_chem_shifts_ppm = np.array([-3.80, -3.40, -2.60, -1.94, -0.39, 0.60])
    fF = B0 * gamma * 1e-6 * fat_chem_shifts_ppm

    def prop_fun(w, f, var_w, var_f):
        # This is a direct translation of the original MATLAB prop_fun.
        # Note: The formula used here is non-standard for error propagation. It appears
        # to square the variances of the parameters, which may not be statistically
        # correct. It is implemented this way to ensure the Python output
        # exactly matches the original MATLAB script's output.
        term1 = (var_w * f / ((w + f)**2))**2
        term2 = (var_f * w / ((w + f)**2))**2
        return 100.0 * np.sqrt(term1 + term2)


    # --- 2. Construct Signal Model and Derivatives ---

    t = t.reshape(N, 1)
    FF, T = np.meshgrid(fF, t)
    FA, _ = np.meshgrid(rel_amps, t)

    AM = np.zeros((N, 2, 4), dtype=np.complex128)
    phase_term = np.exp(1j * phi)
    
    A0 = np.exp(-r2 * t) * phase_term
    A1 = np.sum(FA * np.exp(-r2 * T) * phase_term * np.exp(1j * 2 * np.pi * FF * T), axis=1, keepdims=True)
    AM[:, :, 0] = np.hstack([A0, A1])

    dA0_dr2 = -t * A0
    dA1_dr2 = np.sum(-T * FA * np.exp(-r2 * T) * phase_term * np.exp(1j * 2 * np.pi * FF * T), axis=1, keepdims=True)
    AM[:, :, 1] = np.hstack([dA0_dr2, dA1_dr2])

    dA0_dphi = 1j * A0
    dA1_dphi = 1j * A1
    AM[:, :, 2] = np.hstack([dA0_dphi, dA1_dphi])

    dA0_dfm = 1j * 2 * np.pi * t * A0
    dA1_dfm = np.sum(1j * 2 * np.pi * T * FA * np.exp(-r2 * T) * phase_term * np.exp(1j * 2 * np.pi * FF * T), axis=1, keepdims=True)
    AM[:, :, 3] = np.hstack([dA0_dfm, dA1_dfm])


    # --- 3. Construct the Fisher Information Matrix (FIM) ---
    
    F = np.zeros((5, 5), dtype=np.complex128)
    A = AM[:, :, 0]
    F[0:2, 0:2] = A.conj().T @ A
    
    for ii in range(2, 5):
        for jj in range(2, 5):
            d_ii = AM[:, :, ii-1]
            d_jj = AM[:, :, jj-1]
            F[ii, jj] = (a @ d_ii.conj().T @ d_jj @ a.T)[0,0]

    for jj in range(2, 5):
        d_jj = AM[:, :, jj-1]
        block = A.conj().T @ d_jj @ a.T
        F[0:2, jj] = block.flatten()
        F[jj, 0:2] = block.conj().T.flatten()
    # --- 4. Calculate CRLB from the FIM ---
    
    F_real = np.real(F)
    # Check for singularity before inverting
    if np.linalg.matrix_rank(F_real) < F_real.shape[0]:
        print("WARNING: Fisher Information Matrix is singular. Using pseudo-inverse.")
        inv_F = np.linalg.pinv(F_real)
    else:
        inv_F = np.linalg.inv(F_real)
        
    C = sigma**2 * inv_F


    var_R2s = C[2, 2]
    var_rho_W = C[0, 0]
    var_rho_F = C[1, 1]
    
    # Call the propagation function with the same inputs as MATLAB
    std_dev_F_percent = prop_fun(rho_W, rho_F, var_rho_W, var_rho_F)
    
    return std_dev_F_percent, var_R2s

