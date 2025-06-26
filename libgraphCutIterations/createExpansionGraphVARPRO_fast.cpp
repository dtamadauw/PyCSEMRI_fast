#include "createExpansionGraphVARPRO_fast.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>


using MatrixXXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVectorXd = Eigen::Matrix<double, 1, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;

struct SortPair {
    double ind;
    double val;
};

bool comparePairs(const SortPair& a, const SortPair& b) {
    return a.ind < b.ind;
}

// Helper function to flatten a matrix in Fortran (column-major) order
VectorXd flatten_F(const MatrixXXd& mat, int rows, int cols) {
    VectorXd vec(rows * cols);
    int k = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            vec(k++) = mat(i, j);
        }
    }
    return vec;
}

void save_double_array_to_binary(const std::string& filename, const double* data, size_t num_elements) {
    std::ofstream output_file(filename, std::ios::binary);

    if (!output_file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return;
    }
    size_t total_bytes = num_elements * sizeof(double);
    output_file.write(reinterpret_cast<const char*>(data), total_bytes);
    output_file.close();
}

#ifdef DEBUG_OUTPUT
template<typename T>
void write_debug_matrix(std::ofstream& file, const std::string& name, const T& matrix) {
    file << name << "\n";
    file << matrix.rows() << " " << matrix.cols() << "\n";
    file << std::fixed << std::setprecision(10);
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            file << matrix(i, j) << "\n";
        }
    }
}

template<typename T>
void write_debug_vector(std::ofstream& file, const std::string& name, const std::vector<T>& vec) {
    file << name << "\n";
    file << vec.size() << " " << 1 << "\n";
    file << std::fixed << std::setprecision(10);
    for(const auto& val : vec) {
        file << val << "\n";
    }
}
#endif

// Standalone helper function to replicate Python's C-order ravel_multi_index
// It now correctly mimics Python's behavior of not adjusting for 1-based indexing.
double ravel_c_style(long long r_one, long long c_one, long long num_cols) {
    return (r_one) * num_cols + (c_one);
}

void createExpansionGraphVARPRO_fast_cpp(
    const double* residual_data, int L, int sx, int sy,
    double dfm_val,
    const double* lambdamap_data,
    int size_clique,
    const double* cur_ind_data,
    const double* step_data,
    std::vector<double>& final_values,
    std::vector<int>& final_rows,
    std::vector<int>& final_cols
) {
    long long s = static_cast<long long>(sx) * sy;
    long long num_nodes = s + 2;
    double maxA = 1e6;

    Eigen::Map<const MatrixXXd> cur_ind_map(cur_ind_data, sx, sy);
    Eigen::Map<const MatrixXXd> step_map(step_data, sx, sy);
    Eigen::Map<const MatrixXXd> lambdamap(lambdamap_data, sx, sy);
    
    MatrixXXd step_ind_mat = cur_ind_map.array() + step_map.array();
    
    // Use helper function for robust Fortran-style flattening
    VectorXd cur_ind1d = flatten_F(cur_ind_map, sx, sy);
    VectorXd step_ind1d = flatten_F(step_ind_mat, sx, sy);
    
    MatrixXXd factor = lambdamap.array() * (dfm_val * dfm_val);

#ifdef DEBUG_OUTPUT
    save_double_array_to_binary("GC_residual_data.bin", residual_data, L*sx*sy);
    save_double_array_to_binary("GC_lambdamap_data.bin", lambdamap_data, sx*sy);
    save_double_array_to_binary("GC_cur_ind_data.bin", cur_ind_data, sx*sy);
    save_double_array_to_binary("GC_step_data.bin", step_data, sx*sy);
#endif
    // Corrected, manual Fortran-style flattening based on user feedback
    VectorXd factor1d = flatten_F(factor, sx, sy);
    /*
    VectorXd factor1d(s);
    int k = 0;
    for (int j = 0; j < sy; ++j) { // Iterate over columns first
        for (int i = 0; i < sx; ++i) {
            factor1d(k++) = factor(j, i);
        }
    }
    */

    RowVectorXd valsh = RowVectorXd::Zero(s);
    VectorXd valsv = VectorXd::Zero(s);
    
    std::vector<double> allIndCross;
    std::vector<double> allValsCross;

    MatrixXXd X(sx, sy), Y(sx, sy);
    for(int i=0; i<sx; ++i) for(int j=0; j<sy; ++j) { X(i,j) = i + 1; Y(i,j) = j + 1; }

#ifdef DEBUG_OUTPUT
    std::ofstream loop_debug_file("cpp_loop_debug.txt");
#endif
    // --------------------------------------------------------------------
    // 2. Main Loop: Calculate Pairwise Clique Costs
    // --------------------------------------------------------------------
    for (int dx = -size_clique; dx <= size_clique; ++dx) {
        for (int dy = -size_clique; dy <= size_clique; ++dy) {
            double dist_sq = static_cast<double>(dx * dx + dy * dy);
            if (dist_sq == 0) continue;
            double dist_inv = 1.0 / std::sqrt(dist_sq);

            std::vector<bool> validmapi_mask(s, false);
            std::vector<bool> validmapj_mask(s, false);
            for(int r=0; r<sx; ++r) {
                for(int c=0; c<sy; ++c) {
                    long long flat_idx = c * sx + r;
                    if (X(r, c) + dx >= 1 && X(r, c) + dx <= sx && Y(r, c) + dy >= 1 && Y(r, c) + dy <= sy) {
                        validmapi_mask[flat_idx] = true;
                    }
                    if (X(r, c) - dx >= 1 && X(r, c) - dx <= sx && Y(r, c) - dy >= 1 && Y(r, c) - dy <= sy) {
                        validmapj_mask[flat_idx] = true;
                    }
                }
            }
            
            std::vector<double> factor_i, factor_j, cur_i, cur_j, step_i, step_j;
            std::vector<long long> Sh, Sv; 

            for(long long k=0; k<s; ++k) if(validmapi_mask[k]) {
                factor_i.push_back(factor1d(k)); cur_i.push_back(cur_ind1d(k));
                step_i.push_back(step_ind1d(k)); Sh.push_back(k + 1);
            }
            for(long long k=0; k<s; ++k) if(validmapj_mask[k]) {
                factor_j.push_back(factor1d(k)); cur_j.push_back(cur_ind1d(k));
                step_j.push_back(step_ind1d(k)); Sv.push_back(k + 1);
            }
            
            if (factor_i.size() != factor_j.size()) continue;

            VectorXd a(factor_i.size()), b(factor_i.size()), c(factor_i.size()), d(factor_i.size());
            for(size_t k=0; k<factor_i.size(); ++k){
                double curfactor = std::min(factor_i[k], factor_j[k]);
                a(k) = curfactor * dist_inv * std::pow(cur_i[k] - cur_j[k], 2);
                b(k) = curfactor * dist_inv * std::pow(cur_i[k] - step_j[k], 2);
                c(k) = curfactor * dist_inv * std::pow(step_i[k] - cur_j[k], 2);
                d(k) = curfactor * dist_inv * std::pow(step_i[k] - step_j[k], 2);
            }

            RowVectorXd t1_sh = RowVectorXd::Zero(s);
            for(size_t k=0, i_idx=0; k<s; ++k) if(validmapi_mask[k]) t1_sh(k) = std::max(0.0, c(i_idx) - a(i_idx++));
            valsh += t1_sh;

            VectorXd t1_sv = VectorXd::Zero(s);
            for(size_t k=0, i_idx=0; k<s; ++k) if(validmapi_mask[k]) t1_sv(k) = std::max(0.0, a(i_idx) - c(i_idx++));
            valsv += t1_sv;

            RowVectorXd t2_sh = RowVectorXd::Zero(s);
            for(size_t k=0, j_idx=0; k<s; ++k) if(validmapj_mask[k]) t2_sh(k) = std::max(0.0, d(j_idx) - c(j_idx++));
            valsh += t2_sh;
            
            VectorXd t2_sv = VectorXd::Zero(s);
            for(size_t k=0, j_idx=0; k<s; ++k) if(validmapj_mask[k]) t2_sv(k) = std::max(0.0, c(j_idx) - d(j_idx++));
            valsv += t2_sv;

            VectorXd temp_cross = b + c - a - d;
            //auto ravel_c_idx = [&](long long r_one, long long c_one, long long num_cols) { return (r_one - 1) * num_cols + (c_one - 1); };
            
            std::vector<double> indcross_this_iteration;
            for(size_t k=0; k<Sh.size(); ++k) {
                double flat_idx = ravel_c_style(Sh[k], Sv[k], num_nodes);
                indcross_this_iteration.push_back(flat_idx);
            }
            allIndCross.insert(allIndCross.end(), indcross_this_iteration.begin(), indcross_this_iteration.end());
            allValsCross.insert(allValsCross.end(), temp_cross.data(), temp_cross.data() + temp_cross.size());
            

            #ifdef DEBUG_OUTPUT
                write_debug_matrix(loop_debug_file, "cpp_a_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), a);
                write_debug_matrix(loop_debug_file, "cpp_c_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), c);
                write_debug_matrix(loop_debug_file, "cpp_d_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), d);
                write_debug_vector(loop_debug_file, "cpp_Sh_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), Sh);
                write_debug_vector(loop_debug_file, "cpp_Sv_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), Sv);
                write_debug_vector(loop_debug_file, "cpp_indcross_dx" + std::to_string(dx) + "_dy" + std::to_string(dy), indcross_this_iteration);
                std::ofstream loop_num_nodes_file("cpp_loop_num_nodes_debug.txt");
                loop_num_nodes_file << "num_nodes_check\n1 1\n" << num_nodes << "\n"; // Add this line
                loop_num_nodes_file.close();
            #endif
        }
    }
#ifdef DEBUG_OUTPUT
    loop_debug_file.close();
#endif
    
    VectorXd offset(s);
    for(int i=0; i<s; ++i) offset(i) = i * L;

    Eigen::TensorMap<const Eigen::Tensor<double, 3, Eigen::RowMajor>> residual_tensor(residual_data, L, sx, sy);
    VectorXd residual1D(L*s);
    int k = 0;
    for (int j = 0; j < sy; ++j) {
        for (int i = 0; i < sx; ++i) {
            for (int l = 0; l < L; ++l) {
                residual1D(k++) = residual_tensor(l, i, j);
            }
        }
    }

    VectorXd temp0(s);
    for(int i=0; i<s; ++i) temp0(i) = residual1D(static_cast<int>(cur_ind1d(i) + offset(i)));

    double curmaxA_val = temp0.maxCoeff();
    if (valsh.size() > 0) curmaxA_val = std::max(curmaxA_val, valsh.maxCoeff());
    if (valsv.size() > 0) curmaxA_val = std::max(curmaxA_val, valsv.maxCoeff());
    if(!allValsCross.empty()) {
        double max_cross = 0;
        for(double v : allValsCross) max_cross = std::max(max_cross, v);
        curmaxA_val = std::max(curmaxA_val, max_cross);
    }
    
    double infty = curmaxA_val > 0 ? curmaxA_val : 1e6;
    VectorXd temp1(s);
    for(int i=0; i<s; ++i){
        if(step_ind1d(i) >= 1 && step_ind1d(i) <= L){
            temp1(i) = residual1D(static_cast<int>(step_ind1d(i) + offset(i) - 1));
        } else {
            temp1(i) = infty;
        }
    }

    RowVectorXd values1 = valsh.array() + (temp1 - temp0).transpose().array().cwiseMax(0);
    VectorXd values2 = valsv.array() + (temp0 - temp1).array().cwiseMax(0);

    std::vector<double> indAll, valuesAll;
    indAll.reserve(2 * s + allIndCross.size());
    valuesAll.reserve(2 * s + allValsCross.size());
    
    auto ravel_f = [&](long long r_one, long long c_one, long long num_rows) { return (c_one) * num_rows + (r_one); };
    
    for(long long i=0; i<s; ++i) { indAll.push_back(ravel_f(1, i + 1, num_nodes) - 1.0); valuesAll.push_back(values1(i)); }
    for(long long i=0; i<s; ++i) { indAll.push_back(ravel_f(i + 1, num_nodes-1, num_nodes)); valuesAll.push_back(values2(i)); }
    for(size_t i=0; i<allIndCross.size(); ++i) { indAll.push_back(allIndCross[i]); valuesAll.push_back(allValsCross[i]); }
    
    std::vector<SortPair> pairs;
    for(size_t i = 0; i < indAll.size(); ++i) pairs.push_back({indAll[i], valuesAll[i]});
    std::sort(pairs.begin(), pairs.end(), comparePairs);
    
    double scale_factor = curmaxA_val > 0 ? maxA / curmaxA_val : 1.0;
    
    std::vector<Eigen::Triplet<double>> triplets;
    long long total_elements = num_nodes * num_nodes;
    auto unravel_f = [&](long long flat, long long nrows, int& r, int& c){
        long long wrapped_flat = (flat % total_elements + total_elements) % total_elements;
        r = static_cast<int>(wrapped_flat % nrows);
        c = static_cast<int>(wrapped_flat / nrows);
    };
    
    std::vector<double> cpp_indAllSort, cpp_xind, cpp_yind, cpp_p_val;
    
    for(const auto& p : pairs){
        int r, c;
        unravel_f(static_cast<long long>(p.ind), num_nodes, r, c);
        if (r >= 0 && r < num_nodes && c >= 0 && c < num_nodes) {
            triplets.emplace_back(r, c, p.val);
            cpp_xind.push_back(r);
            cpp_yind.push_back(c);
            cpp_p_val.push_back(p.val);
        }
    }

    

    //printf("scale_factor: %lf\n", scale_factor);


    Eigen::SparseMatrix<double> A_intermediate(num_nodes, num_nodes);
    if (!triplets.empty()) {
        A_intermediate.setFromTriplets(triplets.begin(), triplets.end());
    }

    // Step 3: Iterate through the correctly summed matrix, scale, round, and filter.
    for (int k_outer=0; k_outer<A_intermediate.outerSize(); ++k_outer) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(A_intermediate,k_outer); it; ++it) {
            double scaled_val = std::round(it.value() * scale_factor);
            if (scaled_val > 0) {
                final_values.push_back(scaled_val);
                final_rows.push_back(it.row());
                final_cols.push_back(it.col());
            }
        }
    }


    #ifdef DEBUG_OUTPUT
    std::ofstream debug_file("cpp_debug_intermediates.txt");
    write_debug_matrix(debug_file, "cpp_cur_ind1d", cur_ind1d);
    write_debug_matrix(debug_file, "cpp_step_ind1d", step_ind1d);
    write_debug_matrix(debug_file, "cpp_factor1d", factor1d);
    write_debug_matrix(debug_file, "cpp_valsh", valsh);
    write_debug_matrix(debug_file, "cpp_valsv", valsv);
    write_debug_vector(debug_file, "cpp_allIndCross", allIndCross);
    write_debug_vector(debug_file, "cpp_allValsCross", allValsCross);
    write_debug_matrix(debug_file, "cpp_temp0", temp0);
    write_debug_matrix(debug_file, "cpp_temp1", temp1);
    debug_file << "cpp_curmaxA\n1 1\n" << curmaxA_val << "\n";
    write_debug_vector(debug_file, "cpp_indAll", indAll);
    write_debug_vector(debug_file, "cpp_valuesAll", valuesAll);
    // Add specific variables for unravel debugging
    write_debug_vector(debug_file, "cpp_indAllSort", cpp_indAllSort);
    write_debug_vector(debug_file, "cpp_xind", cpp_xind);
    write_debug_vector(debug_file, "cpp_yind", cpp_yind);
    write_debug_vector(debug_file, "cpp_p_val", cpp_p_val);
    write_debug_vector(debug_file, "cpp_final_values", final_values);
    write_debug_vector(debug_file, "cpp_final_rows", final_rows);
    write_debug_vector(debug_file, "cpp_final_cols", final_cols);
    debug_file.close();
#endif


}


void createExpansionGraphVARPRO_fast_c_wrapper(
    const double* residual_data, int L, int sx, int sy,
    double dfm_val,
    const double* lambdamap_data,
    int size_clique,
    const double* cur_ind_data,
    const double* step_data,
    double** out_values, int** out_rows, int** out_cols, int* out_size
) {
    std::vector<double> values;
    std::vector<int> rows;
    std::vector<int> cols;

    createExpansionGraphVARPRO_fast_cpp(
        residual_data, L, sx, sy, dfm_val, lambdamap_data, size_clique, cur_ind_data, step_data,
        values, rows, cols
    );

    *out_size = values.size();
    *out_values = new double[*out_size];
    *out_rows = new int[*out_size];
    *out_cols = new int[*out_size];
    
    std::copy(values.begin(), values.end(), *out_values);
    std::copy(rows.begin(), rows.end(), *out_rows);
    std::copy(cols.begin(), cols.end(), *out_cols);
}

void free_memory_c_wrapper(double* p1, int* p2, int* p3) {
    delete[] p1;
    delete[] p2;
    delete[] p3;
}
