import numpy as np
from scipy.ndimage import binary_opening, binary_closing

def identify_swapped_regions(pdff_map, swap_threshold=70.0, cleanup_mask=True):
    """
    Identifies potential water-fat swap regions using a simple PDFF threshold.

    This function creates a boolean mask where True indicates a pixel
    suspected of being a water-fat swap. It is based on the principle that
    a swap in a low-fat region will result in an erroneously high,
    physiologically implausible PDFF value.

    Args:
        pdff_map (np.ndarray): 
            2D or 3D NumPy array of the PDFF map. Values should be in
            percent (e.g., 0-100).
        swap_threshold (float, optional): 
            The PDFF percentage above which a pixel is flagged as a potential
            swap. Defaults to 70.0.
        cleanup_mask (bool, optional): 
            If True, applies morphological filtering (opening and closing) to
            remove small, isolated noisy pixels from the mask and fill small
            holes. This is recommended. Defaults to True.

    Returns:
        np.ndarray: 
            A boolean array of the same shape as pdff_map, where True
            indicates a suspected water-fat swap.
    """

    #The threshold criateria of 70% is determined based on the below papers
    #Yokoo, Takeshi, et al. "Linearity, bias, and precision of hepatic proton density fat fraction measurements by using MR imaging: a meta-analysis." Radiology 286.2 (2018): 486-498.
    #Tamada, Daiki, et al. "Confidence maps for reliable estimation of proton density fat fraction and R 2* in the liver." Magnetic resonance in medicine 91.5 (2024): 2172-2187.

    print(f"--- Identifying Water-Fat Swaps (Threshold > {swap_threshold}%) ---")
    
    if not isinstance(pdff_map, np.ndarray):
        raise TypeError("pdff_map must be a NumPy array.")

    # 1. Create the initial mask based on the threshold
    swap_mask = pdff_map > swap_threshold
    
    initial_swapped_pixels = np.sum(swap_mask)
    if initial_swapped_pixels == 0:
        print("No pixels found above the swap threshold. Returning an empty mask.")
        return swap_mask
        
    print(f"Found {initial_swapped_pixels} initial pixels above threshold.")

    # 2. (Optional) Clean up the mask to remove noise
    if cleanup_mask:
        print("Applying morphological filtering to clean up the mask...")
        
        # Define a structuring element for the filter.
        # This creates a 3x3 (or 3x3x3 for 3D) neighborhood.
        structure = np.ones([3] * pdff_map.ndim)
        
        # Perform binary opening: Removes small, isolated "salt" noise.
        # This erodes the mask and then dilates it back.
        opened_mask = binary_opening(swap_mask, structure=structure)
        
        # Perform binary closing: Fills small "pepper" holes inside larger regions.
        # This dilates the mask and then erodes it back.
        final_mask = binary_closing(opened_mask, structure=structure)
        
        final_swapped_pixels = np.sum(final_mask)
        print(f"Finished cleanup. Final mask has {final_swapped_pixels} pixels.")
        
        return final_mask
    else:
        return swap_mask
