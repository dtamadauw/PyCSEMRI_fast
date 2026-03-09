import os
import requests
from tqdm import tqdm
import zipfile
import urllib3

# Suppress insecure request warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def download_example_data(url, dest_path, filename):
    """
    Downloads example data from a URL and saves it to a destination path.
    If the file is a ZIP, extracts it and returns the extracted folder path.
    """
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    
    file_path = os.path.join(dest_path, filename)
    
    if not os.path.exists(file_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True, verify=False)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(file_path, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    else:
        print(f"{filename} already exists.")

    # Check if it's a zip file
    if filename.lower().endswith('.zip'):
        extract_path = os.path.join(dest_path, filename[:-4])
        if not os.path.exists(extract_path):
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
        return extract_path
    
    return file_path

def download_all_example_data(dest_path="./data"):
    """
    Downloads all necessary example data files.
    Returns a tuple of (dicom_path, h5_path).
    """
    # UW-Madison Phantoms/Data
    dicom_url = "https://zenodo.org/records/18906534/files/DICOM.zip?download=1"
    h5_url = "https://zenodo.org/records/18906534/files/IdealChanCombData_6369_5.h5?download=1"
    
    dicom_path = download_example_data(dicom_url, dest_path, "DICOM.zip")
    h5_path = download_example_data(h5_url, dest_path, "IdealChanCombData_6369_5.h5")
    
    return dicom_path, h5_path

if __name__ == "__main__":
    download_all_example_data()
