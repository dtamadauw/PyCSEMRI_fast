import os
import numpy as np
import pydicom
from pydicom.uid import generate_uid
from PIL import Image
from scipy.ndimage import uniform_filter
from pycsemri.identify_susceptibility_regions import identify_susceptibility_regions
from pycsemri.identify_swapped_regions import identify_swapped_regions



def _get_dicom_header_info(directory_path):
    """
    Reads the first DICOM file in a directory and extracts key header information.

    Args:
        directory_path (str): The path to the directory containing DICOM files.

    Returns:
        dict: A dictionary containing the requested DICOM header values.
              Returns None for tags that are not found.
        str: The path to the DICOM file that was read.
        Returns (None, None) if no DICOM files are found.
    """
    print(f"Searching for DICOM files in: {directory_path}")
    
    # Find the first valid DICOM file in the directory
    dicom_file_path = None
    for filename in sorted(os.listdir(directory_path)):
        # Check for .dcm extension or no extension, as both are common
        if filename.lower().endswith('.dcm') or '.' not in filename:
            potential_path = os.path.join(directory_path, filename)
            try:
                # Try to read the file to confirm it's a valid DICOM
                pydicom.dcmread(potential_path, stop_before_pixels=True)
                dicom_file_path = potential_path
                print(f"Found and using DICOM file: {filename}")
                break  # Stop after finding the first valid file
            except pydicom.errors.InvalidDicomError:
                # This file is not a DICOM, continue searching
                continue
    
    if not dicom_file_path:
        print("Error: No valid DICOM files found in the specified directory.")
        return None, None

    # Read the full DICOM header
    try:
        ds = pydicom.dcmread(dicom_file_path)
        
        # Safely extract the required tags using the .get() method
        header_info = {
            'SliceThickness': ds.get('SliceThickness', None),
            'ReconstructionDiameter': ds.get('ReconstructionDiameter', None),
            'Rows': ds.get('Rows', None),
            'Columns': ds.get('Columns', None)
        }
        
        # Note: PixelSpacing can be used to calculate resolution if needed
        pixel_spacing = ds.get('PixelSpacing', [None, None])
        header_info['PixelSpacing_X'] = pixel_spacing[0]
        header_info['PixelSpacing_Y'] = pixel_spacing[1]

        return header_info, dicom_file_path

    except Exception as e:
        print(f"An error occurred while reading the DICOM file: {e}")
        return None, None


def _create_composite_overlay_image(base_map_slice, overlays, overlay_alpha=0.2):
    """
    Creates an RGB image with multiple, multi-colored translucent overlays.

    Args:
        base_map_slice (np.ndarray): The 2D grayscale base image slice.
        overlays (list of tuples): A list where each tuple contains
            (mask_slice, color_rgb). The overlays are applied in order.
        overlay_alpha (float): The transparency of the overlays (0.0 to 1.0).

    Returns:
        np.ndarray: The resulting 3D RGB image array of type uint8.
    """
    # Normalize the base map to 0-255 range for grayscale display
    map_min = np.min(base_map_slice)
    map_max = np.max(base_map_slice)
    norm_map = (base_map_slice - map_min) / (map_max - map_min + 1e-9)
    norm_map_uint8 = (norm_map * 255).astype(np.uint8)

    # Convert the grayscale map to the final RGB image
    final_image = np.stack([norm_map_uint8] * 3, axis=-1)

    # Apply each overlay in the specified order
    for mask_slice, color in overlays:
        if np.any(mask_slice):
            # Create a colored overlay layer for the current mask
            overlay_layer = np.zeros_like(final_image)
            overlay_layer[mask_slice] = color

            # Blend the current state of the image with the new overlay
            blended_map = (overlay_alpha * overlay_layer + (1 - overlay_alpha) * final_image).astype(np.uint8)
            
            # Apply the blended color only where the current mask is true
            final_image[mask_slice] = blended_map[mask_slice]

    return final_image

def _process_volume(base_map, snr_map, required_snr, swap_mask, susceptibility_mask, 
                    input_dir, output_dir, map_type, susc_color, new_seno):
    """
    Helper function to process a 3D volume, creating both overlay and base map DICOMs.
    """
    print(f"\n--- Processing {map_type} Volume ---")
    
    # Create subdirectories for the two output series
    normalized_path = output_dir.rstrip(os.sep)
    parent_dir, final_dir = os.path.split(normalized_path)
    new_final_dir = f"{final_dir}_CM"
    cm_output_dir = os.path.join(parent_dir, new_final_dir)
    base_output_dir = output_dir
    os.makedirs(cm_output_dir, exist_ok=True)
    os.makedirs(base_output_dir, exist_ok=True)

    # Get and sort DICOM files by InstanceNumber
    try:
        dicom_files_with_instance = []
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.dcm', '')):
                dcm_path = os.path.join(input_dir, filename)
                ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                instance_num = ds.get("InstanceNumber", 0)
                dicom_files_with_instance.append((instance_num, dcm_path))
        
        dicom_files_with_instance.sort()
        sorted_dicom_paths = [path for _, path in dicom_files_with_instance]
    except Exception as e:
        print(f"Error reading or sorting DICOMs in {input_dir}: {e}")
        return

    num_slices = base_map.shape[2]
    if len(sorted_dicom_paths) != num_slices:
        print(f"Error: Mismatch between numpy slices ({num_slices}) and DICOM files ({len(sorted_dicom_paths)}).")
        return

    print(f"Found {len(sorted_dicom_paths)} DICOM files, matching {num_slices} array slices.")

    # Generate UIDs for the two new series
    cm_series_uid = generate_uid()
    base_series_uid = generate_uid()
    try:
        first_slice_ds = pydicom.dcmread(sorted_dicom_paths[0], stop_before_pixels=True)
        frame_of_reference_uid = first_slice_ds.get("FrameOfReferenceUID", generate_uid())
    except Exception as e:
        print(f"Could not read FrameOfReferenceUID. Generating a new one. Error: {e}")
        frame_of_reference_uid = generate_uid()

    for z in range(num_slices):
        dcm_path = sorted_dicom_paths[z]
        filename = os.path.basename(dcm_path)
        
        try:
            base_slice = base_map[:, :, z]
            ds_template = pydicom.dcmread(dcm_path)
            original_desc = ds_template.get("SeriesDescription", "Unknown")
            original_series_num = ds_template.get("SeriesNumber", 0)

            # --- 1. Create and save the Confidence Map DICOM ---
            ds_cm = ds_template.copy()
            snr_slice = snr_map[:, :, z]
            required_snr_slice = required_snr[:, :, z]
            swap_slice = swap_mask[:, :, z]
            susc_slice = susceptibility_mask[:, :, z]
            
            overlays = [
                (susc_slice, susc_color),
                (snr_slice <= required_snr_slice, [255, 0, 0]),
                (swap_slice, [255, 255, 0]),
            ]
            overlay_image = _create_composite_overlay_image(base_slice, overlays)

            ds_cm.SeriesInstanceUID = cm_series_uid
            ds_cm.FrameOfReferenceUID = frame_of_reference_uid
            ds_cm.SOPInstanceUID = generate_uid()
            ds_cm.SeriesDescription = f"{original_desc} ConfidenceMap"
            ds_cm.SeriesNumber = original_series_num * 1000 + new_seno + 1
            
            ds_cm.SamplesPerPixel = 3
            ds_cm.PhotometricInterpretation = "RGB"
            ds_cm.PlanarConfiguration = 0
            ds_cm.BitsAllocated = 8
            ds_cm.BitsStored = 8
            ds_cm.HighBit = 7
            if "PixelRepresentation" in ds_cm: del ds_cm.PixelRepresentation
            if "RescaleSlope" in ds_cm: del ds_cm.RescaleSlope
            if "RescaleIntercept" in ds_cm: del ds_cm.RescaleIntercept
            ds_cm.PixelData = overlay_image.tobytes()
            ds_cm.Rows, ds_cm.Columns, _ = overlay_image.shape
            ds_cm.save_as(os.path.join(cm_output_dir, filename))

            # --- 2. Create and save the Base Map DICOM ---
            ds_base = ds_template.copy()
            
            # Set scaling factor to preserve quantitative values
            scale_factor = 1.0
            scaled_slice = (base_slice * scale_factor).astype(np.uint16)

            ds_base.SeriesInstanceUID = base_series_uid
            ds_base.FrameOfReferenceUID = frame_of_reference_uid
            ds_base.SOPInstanceUID = generate_uid()
            ds_base.SeriesDescription = f"{original_desc} BaseMap"
            ds_base.SeriesNumber = original_series_num * 1000 + new_seno

            ds_base.PhotometricInterpretation = "MONOCHROME2"
            ds_base.PixelData = scaled_slice.tobytes()
            ds_base.save_as(os.path.join(base_output_dir, filename))

        except Exception as e:
            print(f"  Error processing slice {z} ({filename}): {e}")

    print(f"Finished processing {map_type} volume.")


def generate_confidence_map_dicoms(
    ims, pdff, r2s, snr_map, imDataParams,
    required_snr_for_pdff, required_snr_for_r2s,
    dcm_ref_dir, dcm_out_dir,
    smoothing_kernel_size=1
):
    """
    Generates DICOM confidence maps for 3D PDFF and R2* volumes.

    Args:
        pdff (np.ndarray): 3D measured PDFF map (X, Y, Z).
        r2s (np.ndarray): 3D measured R2* map (X, Y, Z).
        snr_map (np.ndarray): 3D estimated SNR map (X, Y, Z).
        required_snr_for_pdff (np.ndarray): 3D required SNR for PDFF (X, Y, Z).
        required_snr_for_r2s (np.ndarray): 3D required SNR for R2* (X, Y, Z).
        swap_mask (np.ndarray): 3D water-fat swapping mask (X, Y, Z).
        field_grad_mask_pdff (np.ndarray): 3D susceptibility mask for PDFF (X, Y, Z).
        field_grad_mask_r2s (np.ndarray): 3D susceptibility mask for R2* (X, Y, Z).
        dcm_pdff_dir (str): Path to the original PDFF DICOM directory.
        dcm_r2s_dir (str): Path to the original R2s DICOM directory.
        dcm_pdff_cm_dir (str): Output path for the PDFF confidence map DICOMs.
        dcm_r2s_cm_dir (str): Output path for the R2* confidence map DICOMs.
        smoothing_kernel_size (int): The size of the square moving average window
            to apply to the SNR and threshold maps. A value of 1 means no smoothing.
    """
    print("--- Starting 3D Confidence Map Generation ---")

    # Validate input shapes
    if not (pdff.shape == r2s.shape == snr_map.shape == 
            required_snr_for_pdff.shape == required_snr_for_r2s.shape):
        print("Error: All input numpy arrays must have the same 3D shape.")
        return

    # Calculate susceptibility map 
    header_data, filepath_read = _get_dicom_header_info(dcm_ref_dir)
    if header_data:
        print("Header Data:", header_data)
        print("Slice Thickness:", header_data['SliceThickness'])

    resolution = [header_data['PixelSpacing_X'], header_data['PixelSpacing_X'], header_data['SliceThickness']]
    field_grad_mask_r2s, field_grad_mask_pdff = identify_susceptibility_regions(ims, r2s, imDataParams, resolution)
    
    # Calculate simple water-fat swapping map
    swap_mask = identify_swapped_regions(pdff, swap_threshold=70.0, cleanup_mask=True)


    # --- Apply smoothing filter if requested ---
    if smoothing_kernel_size and smoothing_kernel_size > 1:
        print(f"Applying {smoothing_kernel_size}x{smoothing_kernel_size} moving average filter...")
        
        def _smooth_map(input_map):
            smoothed_map = np.zeros_like(input_map)
            for z in range(input_map.shape[2]):
                smoothed_map[:, :, z] = uniform_filter(input_map[:, :, z], size=smoothing_kernel_size, mode='nearest')
            return smoothed_map

        snr_map_to_use = _smooth_map(snr_map)
        required_pdff_to_use = _smooth_map(required_snr_for_pdff)
        required_r2s_to_use = _smooth_map(required_snr_for_r2s)
        print("Smoothing complete.")
    else:
        snr_map_to_use = snr_map
        required_pdff_to_use = required_snr_for_pdff
        required_r2s_to_use = required_snr_for_r2s

    # Process PDFF Volume
    _process_volume(
        base_map=pdff,
        snr_map=snr_map_to_use,
        required_snr=required_pdff_to_use,
        swap_mask=swap_mask,
        susceptibility_mask=field_grad_mask_pdff,
        input_dir=dcm_ref_dir,
        output_dir=os.path.join(dcm_out_dir, 'PDFF'),
        map_type="PDFF",
        susc_color=[0, 255, 0],  # Green for PDFF susceptibility
        new_seno=0
    )

    # Process R2* Volume
    _process_volume(
        base_map=r2s,
        snr_map=snr_map_to_use,
        required_snr=required_r2s_to_use,
        swap_mask=swap_mask,
        susceptibility_mask=field_grad_mask_r2s,
        input_dir=dcm_ref_dir,
        output_dir=os.path.join(dcm_out_dir, 'R2s'),
        map_type="R2s",
        susc_color=[0, 0, 255],  # Blue for R2* susceptibility
        new_seno=2
    )

    print("\n--- 3D Confidence Map Generation Finished ---")
