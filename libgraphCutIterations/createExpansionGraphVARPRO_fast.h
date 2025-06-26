#ifndef CREATEEXPANSIONGRAPHVARPRO_FAST_H
#define CREATEEXPANSIONGRAPHVARPRO_FAST_H

#include <vector>

// Forward declaration of the C++ function
void createExpansionGraphVARPRO_fast_cpp(
    const double* residual_data, int L, int sx, int sy,
    double dfm_val,
    const double* lambdamap_data,
    int size_clique,
    const double* cur_ind_data,
    const double* step_data,
    // Outputs
    std::vector<double>& out_values,
    std::vector<int>& out_rows,
    std::vector<int>& out_cols
);

// C-style wrapper to be called from Python
extern "C" {
    void createExpansionGraphVARPRO_fast_c_wrapper(
        // Inputs
        const double* residual_data, int L, int sx, int sy,
        double dfm_val,
        const double* lambdamap_data,
        int size_clique,
        const double* cur_ind_data,
        const double* step_data,
        // Output pointers to be filled
        double** out_values,
        int** out_rows,
        int** out_cols,
        int* out_size
    );

    void free_memory_c_wrapper(double* p1, int* p2, int* p3);
}

#endif // CREATEEXPANSIONGRAPHVARPRO_FAST_H
