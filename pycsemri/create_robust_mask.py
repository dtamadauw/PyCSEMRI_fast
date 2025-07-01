import numpy as np
from scipy import ndimage


def _otsu_threshold_np(image):
    """
    Computes the Otsu threshold for a NumPy array.
    
    Parameters:
    image (numpy.ndarray): Grayscale image. Assumed to be normalized between 0 and 1.
    
    Returns:
    float: The optimal threshold value.
    """
    # To avoid floating point issues, scale to a 256-bin histogram
    # This is a common practice when applying Otsu's method to float images.
    scaled_image = (image * 255).astype(np.uint8)
    pixel_counts, _ = np.histogram(scaled_image, bins=256, range=(0, 256))
    total_pixels = scaled_image.size

    # Calculate the cumulative sum of pixels and intensities for efficient computation
    cum_sum = np.cumsum(pixel_counts * np.arange(256))
    cum_weight = np.cumsum(pixel_counts)

    # Total sum of intensities in the image
    total_sum = cum_sum[-1]

    # Handle the case of a flat image (all pixels have the same value)
    if cum_weight[-1] == 0:
        return 0

    # Calculate between-class variance for all possible thresholds
    # We avoid division by zero by checking if weights are zero
    weight_b = cum_weight
    weight_f = total_pixels - weight_b

    # Mask to handle cases where a class has zero pixels
    valid_mask = (weight_b > 0) & (weight_f > 0)
    
    # If no valid threshold is found, return 0
    if not np.any(valid_mask):
        return 0

    # Calculate means for background and foreground
    mean_b = np.divide(cum_sum, weight_b, where=valid_mask)
    mean_f = np.divide(total_sum - cum_sum, weight_f, where=valid_mask)

    # Calculate between-class variance
    between_class_variance = weight_b * weight_f * ((mean_b - mean_f) ** 2)

    # Find the threshold that maximizes the variance
    # The index of the max variance is the optimal threshold in the 0-255 scale
    optimal_threshold_scaled = np.argmax(between_class_variance[valid_mask])
    
    # Rescale the threshold back to the original image's [0, 1] range
    return optimal_threshold_scaled / 255.0




def create_robust_mask(image, percentile_low=1, percentile_high=99, min_size=10):
    """
    Normalize signal intensity and create a boolean mask identifying pixels with signal in a 2D image array.
    
    Parameters:
    image (numpy.ndarray): 2D array representing the image
    percentile_low (float): Lower percentile for contrast stretching
    percentile_high (float): Upper percentile for contrast stretching
    min_size (int): Minimum size of connected components to keep
    
    Returns:
    numpy.ndarray: Boolean mask where True (1) indicates signal, False (0) indicates no signal or noise
    """
    # Apply median filter to reduce noise
    filtered_image = ndimage.median_filter(image, size=3)
    
    # Perform robust contrast stretching using NumPy
    p_low, p_high = np.percentile(filtered_image, (percentile_low, percentile_high))
    
    # Handle the case where p_low and p_high are the same (flat image)
    if p_low == p_high:
        return np.zeros_like(image, dtype=bool)
        
    img_clipped = np.clip(filtered_image, p_low, p_high)
    normalized_image = (img_clipped - p_low) / (p_high - p_low)
    
    # Perform adaptive thresholding using Otsu's method (NumPy implementation)
    threshold = _otsu_threshold_np(normalized_image)
    mask = normalized_image > threshold
    
    # Remove small objects (noise) using binary opening
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Remove small connected components
    # A more efficient way to remove small objects without a loop
    if num_features > 0:
        component_sizes = np.bincount(labeled.ravel())
        too_small = component_sizes < min_size
        remove_pixel_mask = too_small[labeled]
        mask[remove_pixel_mask] = False
    
    return mask
    
    
