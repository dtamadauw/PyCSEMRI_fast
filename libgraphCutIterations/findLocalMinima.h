#ifndef FINDLOCALMINIMA_H
#define FINDLOCALMINIMA_H

#include <vector>

// Struct to hold pointers to intermediate data for debugging
struct IntermediateResults {
    double* dres;
    double* initial_maxres;
    double* initial_minres;
    double* sumres;
    bool* initial_masksignal;
    std::vector<double*>* temp_steps;
    std::vector<int>* temp_kx;
    std::vector<int>* temp_ky;
};

// The core C++ implementation
void findLocalMinima_cpp(
    const double* residual_data, int L, int sx, int sy, double threshold, bool* input_mask_data,
    bool* masksignal_out_data,
    std::vector<double>& resLocalMinima_out_vec,
    long& resLocalMinima_rows,
    double* numMinimaPerVoxel_out_data,
    bool debug,
    IntermediateResults* intermediates
);

// C-style wrapper for ctypes
extern "C" {
    void findLocalMinima_cpp_debug(
        // Inputs
        const double* residual, int L, int sx, int sy,
        double threshold,
        const bool* masksignal_in, // Can be nullptr
        // Outputs
        bool* masksignal_out,
        double** resLocalMinima_out,
        long* resLocalMinima_dims,
        double* numMinimaPerVoxel_out,
        // Intermediate Debug Outputs
        IntermediateResults* intermediates
    );

    void free_memory(double* ptr);
    void free_bool_memory(bool* ptr);
    void free_intermediates(IntermediateResults* intermediates);
}

#endif // FINDLOCALMINIMA_H
