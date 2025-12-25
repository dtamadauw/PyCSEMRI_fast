import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

def identify_susceptibility_regions(ims, r2s_map, imDataParams, resolution):
    """
    Identifies regions with strong susceptibility artifact by analyzing the
    through-plane field gradient.

    Args:
        ims (np.ndarray): 
            4D complex image data with dimensions (X, Y, Echo, Z).
        r2s_map (np.ndarray): 
            3D R2* map with dimensions (X, Y, Z).
        te (np.ndarray): 
            1D array of echo times in seconds.
        resolution (np.ndarray or list): 
            1D array of spatial resolution [res_x, res_y, res_z] in mm.
        b0 (float): 
            Scanner field strength in Tesla (e.g., 1.5 or 3.0).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - field_grad_mask_R2s (np.ndarray): 3D boolean mask for R2*.
            - field_grad_mask_PDFF (np.ndarray): 3D boolean mask for PDFF.
    """
    print("--- Starting Susceptibility Artifact Identification ---")

    te = imDataParams['TE']
    b0 = imDataParams['FieldStrength']

    # --- Step 1: Calculate a rough field map ---
    # The field map is calculated from the phase difference between the first two echoes.
    if ims.shape[2] < 2 or len(te) < 2:
        raise ValueError("Need at least two echoes to calculate a field map.")
    
    x_dim, y_dim, _, z_dim = ims.shape
    field_map = np.zeros((x_dim, y_dim, z_dim))
    delta_te = te[1] - te[0]

    print(f"Calculating rough field map using delta TE = {delta_te * 1000:.2f} ms...")
    for z in range(z_dim):
        im1 = ims[:, :, 0, z]
        im2 = ims[:, :, 1, z]
        
        # Calculate phase difference and unwrap it to get frequency shift (field map)
        phase_diff = np.angle(im2 * np.conj(im1))
        field_map[:, :, z] = phase_diff / (2 * np.pi * delta_te)

    # --- Step 2: Calculate field gradient along Z ---
    # The gradient is calculated in physical units (Hz/mm).
    print(f"Calculating field gradient along Z-axis (slice thickness = {resolution[2]} mm)...")
    # np.gradient calculates the central difference. The third argument is the spacing.
    _, _, field_gradient_z = np.gradient(field_map, resolution[0], resolution[1], resolution[2])
    
    # We are interested in the magnitude of the gradient
    field_gradient_z_abs = np.abs(field_gradient_z)

    # --- Step 3: Smooth the field gradient ---
    # A small sigma is used for gentle smoothing to reduce noise before thresholding.
    print("Smoothing the field gradient map...")
    smoothed_field_gradient = gaussian_filter(field_gradient_z_abs, sigma=1)

    # --- Step 4: Calculate the R2* dependent threshold ---
    print("Calculating dynamic threshold for R2* mask...")
    p_fit = np.array([1.34671399149939e-07, -0.000285349985333389, 0.214128489945153, 28.5122095480830, 7097.55381821749])
    grad_th_r2s = np.zeros_like(r2s_map)

    for z in range(z_dim):
        # Apply a 9x9 moving average filter to each R2* slice
        r2s_slice = r2s_map[:, :, z]
        r2s_filt_slice = uniform_filter(r2s_slice, size=9, mode='nearest')
        
        # Evaluate the polynomial to get the threshold for this slice
        grad_th_r2s[:, :, z] = 1e-3 * np.polyval(p_fit, r2s_filt_slice)
    
    # --- Step 5: Determine the PDFF threshold based on B0 ---
    if b0 == 3.0:
        grad_th_pdff = 51.0  # Hz/mm
    elif b0 == 1.5:
        grad_th_pdff = 93.0  # Hz/mm
    else:
        # Fallback for other field strengths, can be adjusted
        print(f"Warning: No predefined PDFF threshold for B0={b0}T. Using 3T value.")
        grad_th_pdff = 51.0
    print(f"Using fixed threshold for PDFF mask: {grad_th_pdff} Hz/mm")

    # --- Step 6: Generate final masks ---
    print("Generating final boolean masks...")
    field_grad_mask_r2s = smoothed_field_gradient > grad_th_r2s
    field_grad_mask_pdff = smoothed_field_gradient > grad_th_pdff
    
    print("--- Finished ---")
    return field_grad_mask_r2s, field_grad_mask_pdff

