# PyCSEMRI: Python Water-Fat Separation Package for CSE-MRI

## Overview

PyWaterFat is a Python package for water-fat separation in Chemical Shift Encoding Magnetic Resonance Imaging (CSE-MRI). This package is based on Diego Hernando's MATLAB fat-water toolbox, with modifications to improve performance and compatibility with Python ecosystems. The confidence map function will be integrated in the near future.

#### Key Features:
- Implements complex, mixed, and magnitude fitting algorithms for water-fat separation
- No GSL dependency
- Utilizes Eigen C++ library for fast and efficient computations
- Highly portable and easy to install in various environments, including MRI scanners

## Background

This package builds upon the work presented at the ISMRM Workshop on Fat-Water Separation:
[ISMRM Fat-Water Separation Workshop](https://www.ismrm.org/workshops/FatWater12/data.htm)

#### Advantages of Using Eigen:
- No need to install third-party C++ libraries
- Increased portability and ease of installation in various environments, including MRI scanners

## Installation

This package contains C++ components and requires a compiler for installation from source. However, pre-compiled wheels are provided for common platforms (macOS, Linux), making installation easy via pip.

# From PyPI (Recommended for Users)

If you just want to use the package, you can install it directly from the Python Package Index (PyPI). This method will automatically download the correct pre-compiled version for your system.

```pip install PyCSEMRI```



## Usage

A sample code is provided in `Example_ChanComb_h5.py`. 



