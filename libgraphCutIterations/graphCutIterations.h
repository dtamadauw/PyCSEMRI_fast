#ifndef GRAPHCUTITERATIONS_H
#define GRAPHCUTITERATIONS_H

#include <vector>
#include <complex>

// Main C++ implementation of the graph cut iterations
void graphCutIterations_cpp(
    // Input parameters from Python dictionaries
    int sx, int sy, int num_acqs, int nTE,
    const std::vector<std::complex<double>>& images,
    const std::vector<double>& TE,
    double fieldStrength,
    const std::vector<double>& fat_frequencies,
    double lambda_val,
    const std::vector<double>& range_fm,
    int num_fms,
    int num_iters,
    int size_clique,
    // Input arrays from Python
    const double* residual_data,
    const double* lmap_data,
    const double* cur_ind_data,
    // Output arrays (to be allocated and filled by the function)
    double** fm_out,
    bool** masksignal_out,
    // Debugging
    const char* output_dir
);

// C-style wrapper for ctypes
extern "C" {
    void graphCutIterations_c_wrapper(
        // imDataParams
        int sx, int sy, int num_acqs, int nTE,
        const std::complex<double>* images,
        const double* TE,
        double fieldStrength,
        // algoParams
        const double* fat_frequencies, int num_fat_frequencies,
        double lambda_val,
        const double* range_fm,
        int num_fms,
        int num_iters,
        int size_clique,
        // Other inputs
        const double* residual_data,
        const double* lmap_data,
        const double* cur_ind_data,
        // Outputs
        double** fm_out,
        bool** masksignal_out,
        // Debug
        const char* output_dir
    );

    // Memory freeing function
    void free_memory_cpp(void* ptr);
}

#endif // GRAPHCUTITERATIONS_H

