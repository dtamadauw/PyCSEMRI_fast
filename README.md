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

1. Compile the C++ code:

```cd ./src```
```make```

2. Copy the compiled static library (libuwwfs.so) to your library path.


3. Install Python dependencies:

```pip install -r requirements.txt```




## Usage

A sample code is provided in `Example_ChanComb_h5.py`. 



