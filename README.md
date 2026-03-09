# PyCSEMRI: Portable Python Package for Water-Fat Separation in CSE-MRI

[![License: MPL 2.0](https://img.shields.io/badge/License-MPL%202.0-brightgreen.svg)](https://opensource.org/licenses/MPL-2.0)

## Overview

`PyCSEMRI` is a Python package for water-fat separation in Chemical Shift Encoded MRI (CSE-MRI). It provides fast and robust estimation of proton density fat fraction (PDFF), R2*, and field maps.

**Key features:**

- Includes both **graph-cut** and **LUT-based** (`VARPRO_LUT`) algorithms
- `VARPRO_LUT` achieves ~30% of the computation time of graph-cut with reduced water-fat swapping artifacts (validated on 100 patient cases)
- C++ core using **header-only** libraries (Eigen, Boost) — no system-level dependencies or admin permissions required
- Both **Python** and **MATLAB** interfaces
- Pre-compiled **PyPI wheels** for easy installation, including offline environments (clinical scanners)

## Background

The graph-cut algorithm is the most common approach for water-fat separation due to its robustness, particularly in the liver where off-resonance distributions can be complicated. However, it is computationally costly, and existing toolboxes depend on MATLAB and C++ libraries that require admin permissions — making deployment on clinical scanners (no internet, no root access) very challenging.

`PyCSEMRI` addresses these limitations by introducing the `VARPRO_LUT` algorithm and by using only header-only C++ libraries for maximum portability. This package builds upon the algorithms from the [ISMRM Fat-Water Separation Workshop](https://www.ismrm.org/workshops/FatWater12/data.htm).

## Installation

### From PyPI (Recommended)

```bash
pip install pycsemri
```

### From Source

Requires a C++ compiler and CMake.

```bash
git clone https://github.com/dtamadauw/PyCSEMRI_fast.git
cd PyCSEMRI_fast
pip install .
```

### Offline Installation (Clinical Scanners)

For environments without internet access:

1. On a machine with internet:
   ```bash
   mkdir wheelhouse
   pip download pycsemri -d wheelhouse
   ```
2. Transfer the `wheelhouse` folder to the scanner.
3. Install:
   ```bash
   pip install pycsemri --no-index --find-links=wheelhouse
   ```

## Usage

```python
import numpy as np
from pycsemri.VARPRO_LUT import VARPRO_LUT

# Prepare image data (nx, ny, nTE) and echo times
images = ...  # complex-valued numpy array
tes = np.array([1.2, 2.4, 3.6, 4.8, 6.0, 7.2]) * 1e-3

imDataParams = {
    'images': images,
    'TE': tes,
    'FieldStrength': 3.0,
    'PrecessionIsClockwise': 1
}

algoParams = {
    'SUBSAMPLE': 4,
    'range_fm': [-200, 200],
    'NUM_FMS': 41,
    'range_r2star': [0, 100],
    'NUM_R2STARS': 11,
    'species': [
        {'relAmps': [1.0], 'frequency': [0.0]},  # Water
        {
            'relAmps': [0.087, 0.693, 0.128, 0.004, 0.039, 0.014, 0.035],
            'frequency': [-3.8, -3.4, -2.6, -1.9, -0.5, 0.5, 0.6]
        }  # Fat
    ]
}

results = VARPRO_LUT(imDataParams, algoParams)
```

See the `Example_*.py` files for detailed examples with HDF5 and DICOM data.

## Dependencies

- **Python**: numpy, scipy, pydicom
- **C++ (header-only, bundled automatically)**: Eigen, Boost

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs and submitting pull requests.

## License

Mozilla Public License 2.0 (MPL 2.0). See [LICENSE.md](LICENSE.md).

## Citation

If you use this software, please cite:

```bibtex
@article{Tamada2026,
  author  = {Tamada, Daiki and Hernando, Diego and Reeder, Scott B.},
  title   = {PyCSEMRI: A Portable Python Package for Fast and Robust Water-Fat Separation in Chemical Shift Encoded MRI},
  journal = {Journal of Open Source Software},
  year    = {2026}
}
```

## References

- Hernando D, Kellman P, Haldar JP, Liang ZP. Robust water/fat separation in the presence of large field inhomogeneities using a graph cut algorithm. *Magn Reson Med.* 2010;63(1):79-90. [doi:10.1002/mrm.22177](https://doi.org/10.1002/mrm.22177)
- Hu HH, Börnert P, Hernando D, et al. ISMRM Workshop on Fat-Water Separation. *Magn Reson Med.* 2012;68(2):378-388. [doi:10.1002/mrm.24369](https://doi.org/10.1002/mrm.24369)
- [ISMRM Fat-Water Separation Challenge](https://www.ismrm.org/workshops/FatWater12/data.htm)
