import numpy as np


import numpy as np

def calculate_snr_from_fit(
    imDataParams, outParams,
    percentile_cap=95.0
):
    """
    Analytically estimates SNR from CSE-MRI fitting results, with masking for bad fits.

    This function calculates the SNR and an R-squared goodness-of-fit map. It masks
    the SNR map by setting SNR to 0 in pixels where the R-squared value is below
    a specified threshold, effectively removing artifacts from failed fits.

    Args:
        measured_data (np.ndarray):
            The original multi-echo complex image data.
            Shape: (M, N, ..., E), where E is the number of echoes.
        fitted_signal (np.ndarray):
            The ideal signal reconstructed from the fitted parameters.
            Must have the same shape as measured_data.
        rho_w (np.ndarray):
            The fitted complex water signal amplitude map (at TE=0).
            Shape: (M, N, ...).
        rho_f (np.ndarray):
            The fitted complex fat signal amplitude map (at TE=0).
            Shape: (M, N, ...).
        fieldmap (np.ndarray):
            The fitted B0 field map in Hz.
            Shape: (M, N, ...).
        te (np.ndarray):
            A 1D array of echo times in seconds.
        r_squared_threshold (float, optional):
            The minimum R-squared value for a fit to be considered reliable.
            SNR in pixels below this threshold will be set to 0. Defaults to 0.9.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - snr_map (np.ndarray): The estimated SNR map, masked for bad fits.
            - r_squared_map (np.ndarray): The R-squared goodness-of-fit map.
    """
    
    #Draper, N. R., & Smith, H. (1998). Applied Regression Analysis (3rd ed.). Wiley.
    #Seber, G. A. F., & Wild, C. J. (2003). Nonlinear Regression. Wiley.
    #Henkelman, R. M. (1985). Measurement of signal intensities in the presence of noise in MR images. Medical Physics, 12(2), 232-233.
    #Gudbjartsson, H., & Patz, S. (1995). The Rician distribution of noisy MRI data. Magnetic Resonance in Medicine, 34(6), 910-914.
    #Veraart, J., Novikov, D. S., Christiaens, D., Ades-Aron, B., Sijbers, J., & Fieremans, E. (2016). Denoising of diffusion MRI using random matrix theory. NeuroImage, 142, 394-406.


    measured_data = np.squeeze(imDataParams['images'])
    fitted_signal = outParams['fit_amp']
    rho_w = outParams['water_amp']
    rho_f = outParams['fat_amp']
    te = imDataParams['TE']
    fieldmap = outParams['fm']
  

    # --- Step 1: Demodulate the signal to remove off-resonance effects ---
    if te.ndim != 1:
        raise ValueError("te (echo times) must be a 1D array.")
    num_echoes = len(te)
    if measured_data.shape[-1] != num_echoes:
        raise ValueError("The last dimension of measured_data must match the length of te.")

    fieldmap_reshaped = fieldmap[..., np.newaxis]
    te_reshaped = te.reshape(tuple([1] * fieldmap.ndim) + (num_echoes,))

    phase_demod = np.exp(1j * 2.0 * np.pi * fieldmap_reshaped * te_reshaped)
    image_demod = measured_data * phase_demod
    fit_demod = np.conj(fitted_signal) * phase_demod

    # --- Step 2: Calculate Residual Sum of Squares (RSS) ---
    residuals = image_demod - fit_demod
    residual_sum_of_squares = np.sum(np.abs(residuals)**2, axis=-1)

    # --- Step 3: Estimate Noise Variance from RSS ---
    num_fitted_params = 4
    degrees_of_freedom = num_echoes - num_fitted_params
    if degrees_of_freedom <= 0:
        raise ValueError("Degrees of freedom must be positive.")

    noise_variance = residual_sum_of_squares / degrees_of_freedom
    noise_variance[noise_variance <= 0] = np.finfo(float).eps

    # --- Step 4: Calculate Raw Signal Amplitude and SNR ---
    initial_signal_amplitude = np.abs(rho_w) + np.abs(rho_f)
    noise_std_dev = np.sqrt(noise_variance)
    snr_map = initial_signal_amplitude / noise_std_dev

    # --- Step 5: Apply Percentile-Based Capping to Remove Artifacts ---
    if percentile_cap is not None and 0 < percentile_cap < 100:
        #print(f"Applying {percentile_cap}th percentile capping to SNR map...")
        
        # --- Create a foreground mask using Otsu's method for robustness ---
        try:
            # Otsu's method finds an optimal threshold to separate foreground/background
            otsu_threshold = threshold_otsu(initial_signal_amplitude)
            foreground_mask = initial_signal_amplitude > otsu_threshold
        except Exception as e:
            # Fallback for images where Otsu might fail (e.g., all one value)
            #print(f"Warning: Otsu thresholding failed ({e}). Falling back to mean-based threshold.")
            foreground_mask = initial_signal_amplitude > initial_signal_amplitude.mean()


        if np.any(foreground_mask):
            # Calculate the percentile only from the SNR values within the foreground
            snr_in_foreground = snr_map[foreground_mask]
            snr_threshold = np.percentile(snr_in_foreground, percentile_cap)
            
            # Cap all values in the entire map that are above the threshold
            snr_map[snr_map > snr_threshold] = snr_threshold
            #print(f"SNR values capped at: {snr_threshold:.2f}")
        else:
            print("Warning: No foreground signal detected. Skipping SNR capping.")

    return snr_map


