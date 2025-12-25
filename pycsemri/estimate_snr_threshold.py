import numpy as np
from pycsemri.create_full_crlb_num import create_full_crlb_num
from scipy.interpolate import RectBivariateSpline


def estimate_snr_threshold(imDataParams, criteria):
    """
    Estimates the minimum SNR to achieve a certain criteria of PDFF/R2* measurements.

    Args:
        field_strength (float): Magnetic field strength in Tesla.
        te1 (float): First echo time in seconds.
        delta_te (float): Echo spacing in seconds.
        criteria (list or np.array): A 3-element list/array containing:
            - criteria[0] (float): Max variability of PDFF (e.g., 1.0 for 1%).
            - criteria[1] (float): Max absolute variability of R2* (e.g., 5.0).
            - criteria[2] (float): Max relative variability of R2* (e.g., 0.03 for 3%).

    Returns:
        tuple: A tuple containing:
            - max_snr_r2s (np.ndarray): 2D array of required SNR for R2* criteria.
            - max_snr_pdff (np.ndarray): 2D array of required SNR for PDFF criteria.
            - r2s_range (np.ndarray): The range of R2* values used in the calculation.
            - pdff_range (np.ndarray): The range of PDFF values used in the calculation.
    """

    field_strength = imDataParams['FieldStrength']
    te1 = imDataParams['TE'][0]
    delta_te = imDataParams['TE'][1]-imDataParams['TE'][0]

    # --- 1. Define parameter ranges ---
    rho_f_range = np.arange(0, 1.01, 0.1)
    rho_w_range = 1 - rho_f_range
    pdff_range = rho_f_range * 100 
    
    r2s_range = np.arange(10, 1201, 40)
    
    # SNR values to test, from high to low
    snr_search_range = np.arange(2, 101, 1)[::-1]
    
    # Fixed parameters from the MATLAB script
    phi = 0
    psi = 0 # Corresponds to 'fm' in CreateFullCRLBNum, which is not used
    M0 = 100.0 # Reference signal magnitude

    # Extract criteria
    max_pdff_var = criteria[0]
    max_r2s_abs_var = criteria[1]
    max_r2s_rel_var = criteria[2]

    # Initialize output matrices
    max_snr_r2s = np.zeros((len(r2s_range), len(rho_f_range)))
    max_snr_pdff = np.zeros((len(r2s_range), len(rho_f_range)))

    # --- 2. Iterate through all tissue types and find minimum SNR ---
    for fat_idx, rho_f in enumerate(rho_f_range):
        rho_w = 1 - rho_f
        for r2s_idx, r2s in enumerate(r2s_range):
            
            max_found_r2s = False
            max_found_pdff = False
            snr_idx = 0
            
            # Search from high SNR to low SNR
            while (not max_found_r2s or not max_found_pdff) and snr_idx < len(snr_search_range):
                current_snr = snr_search_range[snr_idx]
                
                sigma = M0 / current_snr
                
                norm_rho_w = M0 * rho_w
                norm_rho_f = M0 * rho_f
                
                pdff_std_dev, r2s_var = create_full_crlb_num(
                    norm_rho_w, norm_rho_f, phi, r2s, te1, delta_te, field_strength, 6, sigma
                )

                # --- Check R2* criterion ---
                r2s_criterion = max(max_r2s_abs_var, max_r2s_rel_var * r2s)
                r2s_std_dev = np.sqrt(r2s_var)
                
                if (r2s_std_dev < r2s_criterion * 10) and not max_found_r2s:
                    max_snr_r2s[r2s_idx, fat_idx] = current_snr
                else:
                    max_found_r2s = True

                # --- Check PDFF criterion ---
                pdff_val_to_check = np.sqrt(pdff_std_dev)
                
                if (pdff_val_to_check <= max_pdff_var * 10) and not max_found_pdff:
                    max_snr_pdff[r2s_idx, fat_idx] = current_snr
                else:
                    max_found_pdff = True
                
                snr_idx += 1
                
    max_snr_r2s_mdl = RectBivariateSpline(r2s_range, rho_f_range, max_snr_r2s)
    max_snr_pdff_mdl = RectBivariateSpline(r2s_range, rho_f_range, max_snr_pdff)


    return max_snr_r2s_mdl, max_snr_pdff_mdl, r2s_range, pdff_range

