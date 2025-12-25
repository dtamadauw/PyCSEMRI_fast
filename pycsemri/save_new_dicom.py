import os
import numpy as np
import pydicom
from pydicom.uid import generate_uid

def save_new_dicom(new_image_data, reference_dicom_dir, output_dir, new_seno_offset, new_series_desc="NON_DIAGNOSTIC"):
    """
    Saves a numpy array as a DICOM series using original files as templates.
    
    Args:
        new_image_data (np.array): 3D numpy array [z, y, x] containing the new image.
        reference_dicom_dir (str): Path to directory containing original DICOM files.
        output_dir (str): Path where new DICOMs will be saved.
        new_series_desc (str): Description for the new series.
    """
    
    # 1. Load reference DICOMs
    ref_files = []
    for filename in os.listdir(reference_dicom_dir):
        if filename.endswith(".dcm") or "." not in filename: # Adjust extension check as needed
            filepath = os.path.join(reference_dicom_dir, filename)
            try:
                ds = pydicom.dcmread(filepath)
                # Check if file has pixel data (skip directory files/DICOMDIR)
                if hasattr(ds, 'PixelData'):
                    ref_files.append(ds)
            except:
                continue
    
    if not ref_files:
        raise ValueError("No valid DICOM files found in reference directory.")

    # 2. Sort reference files by slice location (Z-position)
    # This ensures your numpy array [0] maps to the physical top/bottom slice correctly.
    # We sort by ImagePositionPatient Z coordinate.
    ref_files.sort(key=lambda x: x.ImagePositionPatient[2])
    
    # Verify shapes match
    num_slices_ref = len(ref_files)
    num_slices_new = new_image_data.shape[2]
    
    if num_slices_ref != num_slices_new:
        raise ValueError(f"Shape mismatch! Reference has {num_slices_ref} slices, "
                         f"but new data has {num_slices_new} slices.")

    # 3. Generate a new Series Instance UID
    # This links all slices together into one new volume.
    new_series_uid = generate_uid()
    
    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # 4. Iterate and Save
    print(f"Processing {num_slices_new} slices...")
    
    for i, ds in enumerate(ref_files):
        # -- A. Handle Pixel Data --
        # Get the slice from your new data
        slice_data = new_image_data[:, :, i]
        
        # IMPORTANT: DICOM usually requires integer data.
        # If your recon is float, you must normalize/scale it.
        # Here we assume the data is already scaled to uint16 range (0-65535).
        # If not, uncomment the normalization logic below.
        
        # --- Optional Normalization ---
        # slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 65535
        # ------------------------------

        # Convert to uint16
        pixel_array = slice_data.astype(np.uint16)
        
        # Update header to reflect new rows/cols if resolution changed
        ds.Rows = pixel_array.shape[0]
        ds.Columns = pixel_array.shape[1]
        ds.PixelData = pixel_array.tobytes()

        # -- B. Update UIDs and Description --
        ds.SeriesInstanceUID = new_series_uid
        ds.SOPInstanceUID = generate_uid() # Must be unique for EVERY slice
        ds.SeriesDescription = new_series_desc
        ds.SeriesNumber = ds.SeriesNumber + new_seno_offset # Offset series number to separate from original
        
        # Remove checksums/signatures that are now invalid
        if 'PixelData' in ds:
             ds['PixelData'].VR = 'OW' # Explicitly set VR if needed

        # -- C. Save --
        out_name = f"PYCSEMRI_SE{ds.SeriesNumber:04d}_{i:03d}.dcm"
        ds.save_as(os.path.join(output_dir, out_name))

    print(f"Successfully saved {num_slices_new} files to {output_dir}")

# --- Example Usage ---
if __name__ == "__main__":
    # Create dummy data for demonstration (30 slices, 256x256)
    # In reality, this would be your reconstruction variable
    dummy_recon = np.random.randint(0, 65535, (30, 256, 256), dtype=np.uint16)
    
    # Paths (Update these)
    original_dcm_path = "/path/to/original/dicoms"
    output_dcm_path = "/path/to/save/new_dicoms"
    
    try:
        save_new_dicom(dummy_recon, original_dcm_path, output_dcm_path)
    except Exception as e:
        print(f"Error: {e}")