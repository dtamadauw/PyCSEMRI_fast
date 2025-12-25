#ifndef _H_FAST_WFS_
#define _H_FAST_WFS_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream> // For writing debug files
#include <chrono>  // For timing
#include <string.h>
#include <iomanip>
#include <cmath> // For M_PI, std::sqrt, etc.
#include <array> // For optimized median filter

#include "param_str.h"

// --- DEBUG MACRO ---
// Uncomment this line (or pass -DFAST_WFS_DEBUG) to enable debug file output
// #define FAST_WFS_DEBUG
// -------------------


#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif
#define GYRO 42.58
#define MAX_ITER 100

// --- Main function exposed to C/C++ linker ---
#ifdef __cplusplus
extern "C" {
#endif

void VARPRO_LUT(imDataParams_str *imDataParams, algoParams_str *algoParams, outInitParams_str* outInitParams);

#ifdef __cplusplus
}
#endif


// --- Core C++ internal function signatures ---
// (These are not exposed to the C linker)

// --- MODIFICATION: Signature changed to accept two datasets (fast and planned) ---
static void fast_ideal_initial_guess_cpp(
    // Outputs (from 'planned' LR data)
    Eigen::VectorXd& fm_init, 
    Eigen::VectorXd& r2s_init,
    // "Planned" LR Dataset
    const Eigen::MatrixXcd& s_q_vec_lr,
    const Eigen::VectorXd& mask_vec_lr,
    int nx_lr, int ny_lr,
    long long q_masked_lr,
    // "Fast" LR Dataset (for iter 0, 1)
    const Eigen::MatrixXcd& s_q_vec_fast,
    const Eigen::VectorXd& mask_vec_fast,
    int nx_fast, int ny_fast,
    long long q_masked_fast,
    // Common Inputs
    const Eigen::VectorXd& t_n,
    const algoParams_str& algoParams,
    const imDataParams_str& imDataParams,
    // Pre-computed Bases (from Stage 1)
    const Eigen::VectorXcd& b_fat,    // [N x 1]
    const Eigen::VectorXd& absB2,    // [N x 1]
    const Eigen::MatrixXcd& P_coarse, // [N x F]
    const Eigen::MatrixXcd& Pbf,    // [N x F]
    const Eigen::MatrixXd& D_r2_lut, // [N x R]
    const Eigen::MatrixXd& D2_lut    // [N x R]
);

// --- MODIFICATION: Signature changed to accept 2D Maps ---
static void stage5_final_amplitudes_cpp(
    // Outputs (as 2D maps)
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& wat_r_amp,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& wat_i_amp,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& fat_r_amp,
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& fat_i_amp,
    // Inputs
    const Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& fm_map,
    const Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& r2s_map,
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    const imDataParams_str* imDataParams, // Pass pointer to get HR signal
    const Eigen::VectorXd& t_n,
    const Eigen::VectorXcd& b_fat
);

// --- NEW: Helper function for downsampling ---
static void create_downsampled_data(
    // Outputs (passed by reference)
    Eigen::MatrixXcd& s_q_vec_out,
    Eigen::VectorXd& mask_vec_out,
    int& nx_out, int& ny_out,
    long long& q_masked_out,
    // Inputs
    const imDataParams_str* imDataParams, // Original HR data
    int factor,
    double mask_threshold
);


// --- Helper function signatures ---

void write_vector_to_file(const Eigen::VectorXd& vec, const std::string& filename);
void write_complex_matrix_to_file(const Eigen::MatrixXcd& mat, const std::string& filename_prefix);
void write_complex_vector_to_file(const Eigen::VectorXcd& vec, const std::string& filename_prefix);

double median(std::vector<double>& v);
double percentile(std::vector<double>& v, double p);
double std_dev(const std::vector<double>& v);

Eigen::VectorXd median_filter_3x3(const Eigen::VectorXd& in_map, int nx, int ny);

// --- COMPILER FIX: Templated to use Eigen::DenseBase ---
template<typename SrcMatrix, typename DstMatrix>
static void smooth_pass_horizontal_inplace(
    const Eigen::DenseBase<SrcMatrix>& src_map_base, 
    Eigen::DenseBase<DstMatrix>& dst_map_base, 
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius);

// --- COMPILER FIX: Templated to use Eigen::DenseBase ---
template<typename SrcMatrix, typename DstMatrix>
static void smooth_pass_vertical_inplace(
    const Eigen::DenseBase<SrcMatrix>& src_map_base, 
    Eigen::DenseBase<DstMatrix>& dst_map_base, 
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius);

static void smooth_with_mask_inplace(
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& map,
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius, int passes);


#endif // _H_FAST_WFS_