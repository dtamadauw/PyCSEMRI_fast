#ifndef DECOMPOSE_GIVEN_FIELD_MAP_AND_DAMPINGS_H
#define DECOMPOSE_GIVEN_FIELD_MAP_AND_DAMPINGS_H

#include <complex>

#ifdef _WIN32
    #define DLLEXPORT __declspec(dllexport)
#else
    #define DLLEXPORT
#endif

extern "C" {
    /**
     * @brief C++ implementation of decomposeGivenFieldMapAndDampings.
     * * This function performs a water-fat decomposition by solving a linear least-squares
     * problem for each pixel based on a given field map and R2* decay rates.
     * All multi-dimensional arrays are expected in flattened, C-style (row-major) order.
     */
    DLLEXPORT void decomposeGivenFieldMapAndDampings_cpp(
        // Input data pointers
        const double* images_real,
        const double* images_imag,
        const double* fieldmap,
        const double* r2starWater,
        const double* r2starFat,
        const double* t,
        const double* deltaF,
        const double* relAmps,
        
        // Input dimensions and parameters
        int sx,
        int sy,
        int C,
        int N,
        int num_relAmps,
        double ampW,
        int precessionIsClockwise,

        // Output data pointers (pre-allocated by caller)
        double* amps_out_real,
        double* amps_out_imag,

        // Debug output pointers (optional, for a single pixel)
        int debug_kx,
        int debug_ky,
        double* B1_out_real,
        double* B1_out_imag,
        double* B_out_real,
        double* B_out_imag,
        double* s_out_real,
        double* s_out_imag
    );
}

#endif // DECOMPOSE_GIVEN_FIELD_MAP_AND_DAMPINGS_H

