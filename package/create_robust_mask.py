import numpy as np
from scipy import ndimage
from skimage import exposure
from skimage.filters import threshold_otsu


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
    
    # Perform robust contrast stretching
    p_low, p_high = np.percentile(filtered_image, (percentile_low, percentile_high))
    normalized_image = exposure.rescale_intensity(filtered_image, in_range=(p_low, p_high))
    
    # Perform adaptive thresholding using Otsu's method
    threshold = threshold_otsu(normalized_image)
    mask = normalized_image > threshold
    
    # Remove small objects (noise)
    mask = ndimage.binary_opening(mask, structure=np.ones((3,3)))
    
    # Label connected components
    labeled, num_features = ndimage.label(mask)
    
    # Remove small connected components
    for i in range(1, num_features+1):
        if np.sum(labeled == i) < min_size:
            mask[labeled == i] = False
    
    return mask



# Example usage:
#image = np.random.rand(100, 100)  # Replace this with your actual image array
#threshold = 0.5  # Adjust this value based on your signal characteristics
#min_size = 10  # Adjust this value based on the minimum expected size of your signal
#robust_signal_mask = create_robust_signal_mask(image, threshold, min_size)

