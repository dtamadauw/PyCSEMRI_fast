---
title: 'PyCSEMRI: A Portable Python Package for Fast and Robust Water-Fat Separation in Chemical Shift Encoded MRI'
tags:
  - Python
  - C++
  - MRI
  - water-fat separation
  - PDFF
  - R2*
authors:
  - name: Daiki Tamada
    orcid: 0000-0003-3615-459X
    affiliation: 1
  - name: Diego Hernando
    orcid: 0000-0002-0016-0317
    affiliation: 1
  - name: Scott B. Reeder
    orcid: 0000-0003-4728-8171
    affiliation: 1
affiliations:
  - name: University of Wisconsin-Madison, WI, USA
    index: 1
date: 8 March 2026
bibliography: paper.bib
---

# Summary

Chemical Shift Encoding Magnetic Resonance Imaging (CSE-MRI) is an established and essential quantitative modality in modern radiology. Primarily, it enables the precise decomposition of water and fat signals to yield accurate measurements of key clinical biomarkers, such as Proton Density Fat Fraction (PDFF) and the effective transverse relaxation rate, $R_2^*$ [@Reeder:2011]. However, deriving a reliable $B_0$ field map remains a persistent technical hurdle. Without robust estimation of this field map, complex off-resonance effects inevitably cause severe water-fat swapping artifacts, which often happen in regions with high B0 or susceptibility gradients, such as the liver or the spine [@Hernando:2010]. To overcome these computational bottlenecks, we have developed `PyCSEMRI`, a highly portable software library engineered for rapid and robust water-fat separation. Designed with clinical versatility in mind, the package offers seamless interoperability through both Python and MATLAB bindings. It offers two solvers: the widely validated graph-cut algorithm [@Hernando:2010], alongside a novel, highly optimized Look-Up Table approach( (`VARPRO_LUT`)). Most importantly, by minimizing reliance on cumbersome external libraries or complex compiled binaries, `PyCSEMRI` is uniquely adapted for deployment in strictly isolated IT architectures. This zero-dependency approach makes it an ideal solution for direct installation on secure, offline clinical MRI scanners/consoles where internet-based package managers are entirely inaccessible.

# Statement of Need

Since Dixon’s original demonstration of chemical shift-based water-fat separation [@Dixon:1984], water/fat imaging techniques have progressed significantly. Today, CSE-MRI is an essential tool for the non-invasive quantification of hepatic steatosis [@Reeder:2011]. Hepatic steatosis is characterized by the abnormal accumulation of fat within hepatocytes and serves as the primary histological feature of non-alcoholic fatty liver disease (NAFLD). Affecting more than 20% of the general population, NAFLD is the most common chronic liver disease in the United States and carries serious health risks [@Reeder:2011]. If left unmonitored, the condition can progress to non-alcoholic steatohepatitis (NASH), fibrosis, cirrhosis, and ultimately, hepatocellular carcinoma.

Currently, liver biopsy is the clinical gold standard for evaluating liver fat. However, this invasive procedure has significant limitations. In addition to carrying bleeding risks, it is subject to spatial sampling errors because a biopsy needle captures only a tiny fraction of the liver. Therefore, biopsy is not suitable for population screening, longitudinal monitoring, or large-scale clinical trials. Furthermore, other non-invasive imaging methods have limited accuracy for fat quantification. For example, ultrasound depends heavily on the operator's skill, and computed tomography (CT) has low sensitivity for detecting mild fat accumulation.

In contrast, MRI-based techniques can effectively separate the acquired liver signal into its water and fat components. This allows for the calculation of the proton density fat-fraction (PDFF), which is a fundamental, standardized, and platform-independent biomarker for liver fat content [@Reeder:2011]. However, accurate PDFF quantification requires robust estimation of the $B_0$ field map and the $R_2^*$ relaxation rate. These estimations are necessary to remove confounding factors and successfully prevent severe water-fat swapping artifacts.




In water-fat separation research, the graph-cut algorithm is widely regarded as the gold standard. This reputation stems from its exceptional robustness when imaging anatomies with complex off-resonance distributions, such as the liver [@Hernando:2010]. Nevertheless, the practical implementation of this approach involves two primary challenges:

1. **Computational Cost.** Optimizing the graph-cut algorithm requires relatively large computational resources. Due to this costly processing, clinical environments frequently rely on faster alternative techniques like region-growing algorithms [@Yu:2005]. While these alternative methods offer lower robustness, they remain necessary to ensure acceptable image reconstruction times during busy patient schedules.

2. **Deployment Barriers.** The ISMRM Fat-Water Toolbox [@Hu:2012; @ISMRM:2012] stands as the most commonly utilized resource in our community. Originally developed by Diego Hernando, this reference software is developed in MATLAB and C++. It depends on several external libraries that usually require administrative or root permission for compilation and installation. This dependency creates a major obstacle for direct integration into clinical MRI scanners. Such hospital systems are governed by exceedingly strict IT security protocols that completely block internet access and forbid any root-level modifications.

`PyCSEMRI` successfully overcomes these critical limitations by combining novel algorithmic strategies with modern software engineering practices:

- **Fast and Robust LUT-Based Algorithm.** To overcome the heavy processing loads associated with graph-cut optimization, `PyCSEMRI` introduces `VARPRO_LUT`. This novel Look-Up Table algorithm provides rapid and stable estimation of both PDFF and $R_2^*$ values. The performance gain is primarily achieved through a highly refined field map estimation process. We evaluated this approach using clinical datasets from 100 patient examinations. During these trials, the LUT method consumed approximately 30% of the processing time required by conventional techniques. Furthermore, it demonstrated a substantial reduction in severe water-fat swapping artifacts near tissue boundaries. The software also retains a complete implementation of the traditional graph-cut algorithm [@Hernando:2010] to support researchers who require established reference methodologies.

- **High Portability and Easy Installation.** Previous software solutions often struggle with complicated compilation prerequisites. In contrast, we engineered `PyCSEMRI` specifically to maximize portability across different systems (OS and chip architecture), such as researcher's laptops, research workstations, and MRI scanners. The computational core leverages C++ for maximum efficiency. It utilizes the Eigen library [@Eigen] to handle intensive matrix operations and incorporates the Boost C++ libraries [@Boost] to execute the graph-cut logic. A critical design choice was integrating both Eigen and Boost exclusively as header-only libraries. This architectural decision completely removes the necessity for linking against troublesome compiled dependencies such as the GNU Scientific Library (GSL).

Furthermore, `PyCSEMRI` is officially distributed through the Python Package Index (PyPI). We provide pre-compiled binary wheels for major operating systems, including macOS and various Linux distributions. This distribution strategy allows end users to skip the compilation phase entirely. Radiologists and imaging scientists can install the software effortlessly using a simple `pip install pycsemri` command. This simplified process works perfectly even within strictly offline computing environments that lack administrator privileges, such as the reconstruction computers attached to clinical MRI scanners. Finally, the tool offers dedicated programming interfaces for both Python and MATLAB. By supporting both modern open-source data pipelines and established legacy workflows, `PyCSEMRI` effectively bridges the divide between cutting-edge quantitative imaging research and routine hospital deployment.


# State of the Field

Variable Projection (VARPRO) and graph-cut optimization represent the mathematical gold standards for water-fat separation [@Hernando:2010; @Hu:2012]. While the graph-cut technique is highly valued for its spatial robustness, its immense computational burden remains a significant drawback. Consequently, commercial scanner vendors have frequently favored faster but inherently less reliable region-growing algorithms to meet strict clinical time constraints [@Yu:2005].

Recently, deep learning architectures such as U-Nets and conditional Generative Adversarial Networks (cGANs) have gained traction as viable alternatives. These neural networks can execute water-fat separation without relying on the complex initialization steps required by traditional optimization routines [@Basty:2023; @Jafari:2021]. However, conventionally trained supervised models frequently struggle with poor generalizability. They often yield suboptimal image quality or fail completely when encountering datasets with altered acquisition parameters, distinct field strengths like 1.5T or 3.0T, or different echo train lengths than those present in their original training cohorts [@Ganeshkumar:2025; @Jafari:2021]. To mitigate these generalization failures, researchers have proposed physics-informed unsupervised algorithms and ad-hoc deep learning reconstruction techniques. Strategies such as Deep Complex Convolutional Networks optimize the neural network directly on individual test datasets by incorporating the biophysical signal model into the loss function [@Ganeshkumar:2025; @Jafari:2021]. Despite offering vastly improved flexibility, these customized network approaches introduce severe computational bottlenecks. They routinely require thousands of training epochs and demand several minutes of high-end GPU processing per patient volume. This extreme latency currently prohibits their translation into immediate, real-time clinical workflows [@Ganeshkumar:2025; @Jafari:2021].

Therefore, the community still requires a solution that is simultaneously rapid, highly robust, and effortlessly deployable. `PyCSEMRI` directly addresses this enduring need. By building upon proven optimization frameworks, the package introduces the `VARPRO_LUT` methodology. This novel strategy successfully combines the exceptional processing speed of region-growing algorithms with the renowned spatial reliability of the graph-cut technique. To ensure maximum utility across different research pipelines, the software library supplies both the LUT-based and graph-cut solvers alongside seamless integration for Python and MATLAB environments.


# Software Design

`PyCSEMRI` utilizes a hybrid architecture:

- **C++ Core**: The performance-critical fitting, graph-cut, and `VARPRO_LUT` algorithms are implemented in C++ using header-only libraries: Eigen [@Eigen] for linear algebra and Boost [@Boost] for graph operations. Because these are header-only, no system-level installation or administrative permissions are required.
- **Python Interface**: `ctypes` and `numpy` provide a seamless Pythonic interface, allowing users to process HDF5 or DICOM data with minimal code.
- **MATLAB Interface**: MATLAB wrappers are included for users who prefer the MATLAB environment or need to integrate with existing clinical workflows.
- **Build System**: `scikit-build` and `cmake` manage the compilation process. Pre-compiled wheels are distributed via PyPI for easy installation, including offline deployment using the wheelhouse method.

## VARPRO_LUT Algorithm

The CSE-MRI signal at each echo time $t_n$ follows the model:

$$s(t_n) = \bigl(w + f \cdot b_\text{fat}(t_n)\bigr) \cdot e^{i 2\pi f_m t_n} \cdot e^{-R_2^* t_n}$$

where $w$ and $f$ (complex water and fat amplitudes) are **linear** parameters, while $f_m$ (field map) and $R_2^*$ are **nonlinear** parameters. A naïve brute-force search over all six unknowns would be computationally infeasible. The algorithm applies **Variable Projection (VARPRO)** to analytically eliminate the linear parameters: for any fixed $(f_m, R_2^*)$ candidate, the optimal $(w, f)$ are obtained in closed form by solving a $2 \times 2$ Gram system. This reduces the search space from a six-dimensional grid to a compact two-dimensional Look-Up Table (LUT) of $F \times R$ candidates — the key to computational efficiency.

The `VARPRO_LUT` algorithm proceeds through the following stages:

1. **Multi-resolution downsampling.** Two downsampled datasets are created: a coarse "fast" version (8× downsample) for rapid initial estimation, and a finer "planned" version (e.g. 4×) for the final search.
2. **LUT pre-computation.** Phase modulation, $R_2^*$ decay, and fat basis matrices are pre-computed for all candidate values, avoiding repeated `exp()` evaluations per voxel.
3. **Iterative VARPRO-LUT search.** For each $(f_m, R_2^*)$ candidate, the VARPRO residual (total signal energy minus projection energy) is computed for all voxels simultaneously via matrix operations. Three iterations with progressive re-centering ensure that the fixed-size LUT grid covers the true frequency range of the data, followed by a median filter to remove outlier voxels.
4. **Spatial regularization.** A weighted graph Laplacian is constructed from voxel-wise confidence values (inverse residuals), and a Conjugate Gradient solver enforces spatial smoothness of the field map.
5. **Upsampling and smoothing.** The low-resolution field map and $R_2^*$ are bilinearly interpolated to full resolution and smoothed with a mask-aware Gaussian filter.
6. **Final amplitude estimation.** At full resolution, the VARPRO linear solve is applied: with the regularized $(f_m, R_2^*)$, a complex least-squares decomposition recovers the water and fat amplitude maps.


# Research Impact

The algorithms implemented in `PyCSEMRI` have been used and tested in research projects focused on liver fat quantification. By utilizing the novel Look-Up Table methodology, investigators can evaluate extensive clinical datasets with remarkable efficiency. When compared directly to traditional graph-cut techniques, this optimized approach drastically accelerates the overall image reconstruction pipeline without compromising analytical accuracy.

# Acknowledgements

The authors acknowledge the support of GE Healthcare. This package builds upon the algorithms developed for the ISMRM Fat-Water Separation Challenge [@ISMRM:2012].

# References
