#include "graphCutIterations.h"
#include "createExpansionGraphVARPRO_fast.h"
#include "graph.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <unsupported/Eigen/CXX11/Tensor>

// --- Forward declaration to fix linker error ---
struct IntermediateResults;
void findLocalMinima_cpp(
    const double* residual_data, int L, int sx, int sy, double threshold, const bool* input_mask_data,
    bool* masksignal_out_data,
    std::vector<double>& resLocalMinima_out_vec,
    long& resLocalMinima_rows,
    double* numMinimaPerVoxel_out_data,
    bool debug,
    IntermediateResults* intermediates
);
// --- End of fix ---

using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using MatrixXb = Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorXd = Eigen::VectorXd;
using Tensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;

// This function replicates Python's `some_matrix.flatten(order='F')`
template <typename Derived>
auto flatten_F_style(const Eigen::MatrixBase<Derived>& mat) {
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> vec(mat.size());
    int k = 0;
    for (int j = 0; j < mat.cols(); ++j) {
        for (int i = 0; i < mat.rows(); ++i) {
            vec(k++) = mat(i, j);
        }
    }
    return vec;
}

template <typename Derived>
auto reshape_to_F_style_2D(const Eigen::MatrixBase<Derived>& vec, int rows, int cols) {
    using Scalar = typename Derived::Scalar;
    
    // Ensure the input vector has the correct number of elements
    if (vec.size() != rows * cols) {
        throw std::invalid_argument("Input vector size does not match output matrix dimensions.");
    }

    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);

    int k = 0; // Linear index for the 1D input vector
    for (int j = 0; j < cols; ++j) {     // Iterate through columns first
        for (int i = 0; i < rows; ++i) { // Then iterate through rows
            mat(i, j) = vec(k++);
        }
    }
    
    return mat;
}

template<typename T>
void write_debug_data(const std::string& path, const T& data) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(10);
        file << data;
        file.close();
    } else {
        std::cerr << "Could not open file for writing: " << path << std::endl;
    }
}
void write_debug_data_bool(const std::string& path, const MatrixXb& data) {
    std::ofstream file(path);
    if (file.is_open()) {
        file << data.template cast<int>();
        file.close();
    } else {
        std::cerr << "Could not open file for writing: " << path << std::endl;
    }
}
template<typename T>
void write_debug_data_vec(const std::string& path, const std::vector<T>& data) {
    std::ofstream file(path);
    if (file.is_open()) {
        for(const auto& val : data) {
            file << val << "\n";
        }
        file.close();
    } else {
         std::cerr << "Could not open file for writing: " << path << std::endl;
    }
}
void write_debug_tensor(const std::string& path, const Tensor3d& data) {
    std::ofstream file(path);
     if (file.is_open()) {
        file << std::fixed << std::setprecision(10);
        for(int i=0; i < data.dimension(0); ++i)
            for(int j=0; j < data.dimension(1); ++j)
                for(int k=0; k < data.dimension(2); ++k)
                    file << data(i,j,k) << "\n";
        file.close();
    } else {
         std::cerr << "Could not open file for writing: " << path << std::endl;
    }
}


void graphCutIterations_cpp(
    int sx, int sy, int num_acqs, int nTE,
    const std::vector<std::complex<double>>& images,
    const std::vector<double>& TE,
    double fieldStrength,
    const std::vector<double>& fat_frequencies_in,
    double lambda_val,
    const std::vector<double>& range_fm,
    int num_fms,
    int num_iters,
    int size_clique,
    const double* residual_data,
    const double* lmap_data,
    const double* cur_ind_data_in,
    double** fm_out,
    bool** masksignal_out,
    const char* output_dir_c) {

    std::string output_dir(output_dir_c);
    std::mt19937 gen(0); 
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::normal_distribution<> normal_dis(0.0, 1.0);

    constexpr bool STARTBIG = true;
    constexpr int dkg = 15;
    constexpr bool SMOOTH_NOSIGNAL = true;
    constexpr double gyro = 42.58;
    
    VectorXd deltaF(fat_frequencies_in.size() + 1);
    deltaF(0) = 0.0;
    for (size_t i = 0; i < fat_frequencies_in.size(); ++i) deltaF(i + 1) = gyro * fat_frequencies_in[i] * fieldStrength;
    
    double dt = TE[1] - TE[0];
    double period = 1.0 / dt;
    VectorXd fms = VectorXd::LinSpaced(num_fms, range_fm[0], range_fm[1]);
    double dfm = fms(1) - fms(0);

    Eigen::Map<const MatrixXd> lmap(lmap_data, sx, sy);
    MatrixXd cur_ind = Eigen::Map<const MatrixXd>(cur_ind_data_in, sx, sy);
    
    bool* masksignal_data = new bool[sx * sy];
    std::vector<double> resLocalMinima_vec;
    long resLocalMinima_rows = 0;
    double* numMinimaPerVoxel_data = new double[sx * sy];

    findLocalMinima_cpp(residual_data, num_fms, sx, sy, 0.06, nullptr,
                        masksignal_data, resLocalMinima_vec, resLocalMinima_rows, numMinimaPerVoxel_data, false, nullptr);
    
    Eigen::Map<MatrixXb> masksignal(masksignal_data, sx, sy);
    Eigen::TensorMap<const Tensor3d> resLocalMinima_tensor(resLocalMinima_vec.data(), resLocalMinima_rows, sx, sy);
    Eigen::Map<MatrixXd> numMinimaPerVoxel(numMinimaPerVoxel_data, sx, sy);
    long numLocalMin = resLocalMinima_rows;

    MatrixXd fm = MatrixXd::Zero(sx, sy);
    MatrixXd lambdamap = lmap;
    
    std::string initial_prefix = output_dir + "/cpp_initial";
    // ADDED: Save residual data
    Eigen::TensorMap<const Tensor3d> residual_tensor_map(residual_data, sx, sy, num_fms);
    /*
    write_debug_tensor(initial_prefix + "_residual_data.txt", residual_tensor_map);
    write_debug_data(initial_prefix + "_sx.txt", VectorXd::Constant(1, sx));
    write_debug_data(initial_prefix + "_sy.txt", VectorXd::Constant(1, sy));
    write_debug_data(initial_prefix + "_num_fms.txt", VectorXd::Constant(1, num_fms));
    write_debug_data_bool(initial_prefix + "_masksignal.txt", masksignal);
    write_debug_tensor(initial_prefix + "_resLocalMinima.txt", resLocalMinima_tensor);
    write_debug_data(initial_prefix + "_numMinimaPerVoxel.txt", numMinimaPerVoxel);
    write_debug_data(initial_prefix + "_numLocalMin.txt", VectorXd::Constant(1, numLocalMin));
    write_debug_data(initial_prefix + "_dfm.txt", VectorXd::Constant(1, dfm));
    write_debug_data(initial_prefix + "_deltaF.txt", deltaF);
    write_debug_data(initial_prefix + "_period.txt", VectorXd::Constant(1, period));
    write_debug_data(initial_prefix + "_dt.txt", VectorXd::Constant(1, dt));
    write_debug_data_vec(initial_prefix + "_TE.txt", TE);
        */

    VectorXd stepoffset_vec(sx * sy);
    for(int i=0; i < sx*sy; ++i) stepoffset_vec(i) = i * numLocalMin;
    //write_debug_data(initial_prefix + "_stepoffset.txt", stepoffset_vec);
    
    for (int kg = 1; kg <= num_iters; ++kg) {
        //std::cout << "--- C++: Iteration " << kg << " ---" << std::endl;
        //std::string iter_prefix = output_dir + "/cpp_iter_" + std::to_string(kg);

        double prob_bigJump = 0.5;
        if (kg == 1 && STARTBIG) {
            lambdamap = lambda_val * lmap;
            prob_bigJump = 1.0;
        } else if ((kg == dkg && SMOOTH_NOSIGNAL) || !STARTBIG) {
            lambdamap = lambda_val * lmap;
        }
        //write_debug_data(iter_prefix + "_lambdamap.txt", lambdamap);
        //write_debug_data(iter_prefix + "_cur_ind_start.txt", cur_ind);

        int cur_sign = (kg % 2 == 0) ? 1 : -1;
        //write_debug_data(iter_prefix + "_cur_sign.txt", VectorXd::Constant(1, cur_sign));
        MatrixXd cur_step(sx, sy);

        auto cur_step1d = flatten_F_style(cur_step);
        auto cur_ind1d = flatten_F_style(cur_ind);
        
        //if(kg > 2) prob_bigJump = 0.0;//Debug
        //else prob_bigJump = 1.0;
        
        if (dis(gen) < prob_bigJump) {
             //printf("C++: prob_bigJump TRUE\n");
             Tensor3d repCurInd_tensor(numLocalMin, sx, sy);
             for(long r=0; r<numLocalMin; ++r) {
                 for(int c=0; c<sy; ++c) for(int i=0; i<sx; ++i) {
                     repCurInd_tensor(r,i,c) = cur_ind(i,c);
                 }
             }
             //write_debug_tensor(iter_prefix + "_repCurInd.txt", repCurInd_tensor);

             Eigen::Tensor<bool, 3, Eigen::RowMajor> stepLocator_bool_tensor;
             if(cur_sign > 0) {
                 stepLocator_bool_tensor = (repCurInd_tensor + 20/dfm >= resLocalMinima_tensor) && (resLocalMinima_tensor > 0.0);
             } else {
                 stepLocator_bool_tensor = (repCurInd_tensor - 20/dfm > resLocalMinima_tensor) && (resLocalMinima_tensor > 0.0);
             }
            
            Eigen::array<int, 1> sum_axis = {0};
            Eigen::Tensor<double, 2, Eigen::RowMajor> stepLocator_tensor_sum = stepLocator_bool_tensor.cast<double>().sum(sum_axis);
            MatrixXd stepLocator = Eigen::Map<MatrixXd>(stepLocator_tensor_sum.data(), sx, sy);
            if(cur_sign > 0) stepLocator.array() += 1;
            //write_debug_data(iter_prefix + "_stepLocator.txt", stepLocator);
             
            MatrixXb validStep(sx, sy);
            if(cur_sign > 0) for(int i=0; i<sx*sy; ++i) validStep(i) = masksignal(i) && (stepLocator(i) <= numMinimaPerVoxel(i));
            else for(int i=0; i<sx*sy; ++i) validStep(i) = masksignal(i) && (stepLocator(i) >= 1);
            //write_debug_data_bool(iter_prefix + "_validStep.txt", validStep);
             
            VectorXd resLocalMinima1D(numLocalMin * sx * sy);
            int count = 0;
            for (int j = 0; j < sy; ++j) for (int i = 0; i < sx; ++i) for (long l = 0; l < numLocalMin; ++l) {
                resLocalMinima1D(count++) = resLocalMinima_tensor(l, i, j);
            }
            //write_debug_data(iter_prefix + "_resLocalMinima1D.txt", resLocalMinima1D);

            MatrixXd nextValue = MatrixXd::Zero(sx, sy);
            
            auto nextValue1D = flatten_F_style(nextValue);
            auto stepLocator1D = flatten_F_style(stepLocator);
            auto validStep_flat = flatten_F_style(validStep);
            //auto cur_ind1d = flatten_F_style(cur_ind);

            //write_debug_data(iter_prefix + "_stepLocator1D.txt", stepLocator1D);
            //write_debug_data(iter_prefix + "_validStep_flat.txt", validStep_flat.cast<int>());
            //write_debug_data(iter_prefix + "_cur_ind1d.txt", cur_ind1d);

            for(int i=0; i<sx*sy; ++i) {
                if(validStep_flat(i)) {
                    long next_ind = static_cast<long>(stepoffset_vec(i) + stepLocator1D(i) - 1);
                    if(next_ind >=0 && next_ind < resLocalMinima1D.size()) {
                        nextValue1D(i) = resLocalMinima1D(next_ind);
                    }
                }
            }
            nextValue = reshape_to_F_style_2D(nextValue1D, sx, sy);//Eigen::Map<MatrixXd>(nextValue1D.data(),sx, sy);
            //write_debug_data(iter_prefix + "_nextValue.txt", nextValue);
            //write_debug_data(iter_prefix + "_nextValue1D.txt", nextValue1D);

            //auto cur_step1d = flatten_F_style(cur_step);
            
            double nosignal_jump = (dis(gen) < 0.5) ? cur_sign * round(abs(deltaF(1)) / dfm) : cur_sign * abs(round((period - abs(deltaF(1))) / dfm));
            //write_debug_data(iter_prefix + "_nosignal_jump.txt", VectorXd::Constant(1, nosignal_jump));
            
            for(int i=0; i<sx*sy; ++i) {
                if(validStep_flat(i)) {
                    cur_step1d(i) = nextValue1D(i) - cur_ind1d(i);
                } else {
                    cur_step1d(i) = nosignal_jump;
                }
            }
            cur_step = reshape_to_F_style_2D(cur_step1d, sx, sy);//Eigen::Map<MatrixXd>(cur_step1d.data(), sx, sy);
            //write_debug_data(iter_prefix + "_cur_step1d.txt", cur_step1d);



        } else {
            //printf("C++: prob_bigJump FALSE\n");
            //write_debug_data(iter_prefix + "_cur_sign.txt", VectorXd::Constant(1, cur_sign));
            double rnd_numf = std::ceil(std::abs(normal_dis(gen) * 3.0));
            //printf("rnd_numf = %f\n", rnd_numf);
            double all_jump = cur_sign * rnd_numf;
            cur_step = MatrixXd::Constant(sx, sy, all_jump);
            MatrixXd nextValue = cur_ind + cur_step;
            for(int i=0; i<sx*sy; ++i) {
                if(cur_sign > 0 && nextValue(i) > num_fms) cur_step(i) = num_fms - cur_ind(i);
                else if (cur_sign < 0 && nextValue(i) < 1) cur_step(i) = 1 - cur_ind(i);
            }

            //auto cur_step1d = flatten_F_style(cur_step);
            //auto cur_ind1d = flatten_F_style(cur_ind);
        }
        //write_debug_data(iter_prefix + "_cur_step.txt", cur_step);
        
        std::vector<double> A_values;
        std::vector<int> A_rows, A_cols;
        auto lambdamap1d = flatten_F_style(lambdamap);
        createExpansionGraphVARPRO_fast_cpp(residual_data, num_fms, sx, sy, dfm, lambdamap1d.data(), size_clique, cur_ind1d.data(), cur_step1d.data(), A_values, A_rows, A_cols);
        
        //write_debug_data_vec(iter_prefix + "_A_values.txt", A_values);
        //write_debug_data_vec(iter_prefix + "_A_rows.txt", A_rows);
        //write_debug_data_vec(iter_prefix + "_A_cols.txt", A_cols);
        
        // --- Graph-Cut and Update Logic (Corrected) ---
        // CORRECTED: Use a typedef for clarity and correctness
        typedef Graph<int, int, int> GraphType;
        int num_pixel_nodes = sx * sy;
        int num_graph_nodes = num_pixel_nodes + 2;
        GraphType *g = new GraphType(num_pixel_nodes, A_values.size());
        
        // CORRECTED: Call add_node() in a loop
        g->add_node(num_pixel_nodes);

        for (size_t i = 0; i < A_values.size(); ++i) {
            int u = A_rows[i];
            int v = A_cols[i];
            int capacity = static_cast<int>(A_values[i]);
            if (capacity < 0) capacity = 0;

            if (u == 0) {
                g->add_tweights(v - 1, capacity, 0);
            } else if (v == num_graph_nodes - 1) {
                g->add_tweights(u - 1, 0, capacity);
            } else {
                g->add_edge(u - 1, v - 1, capacity, 0);
            }
        }
        
        g->maxflow();
        
        std::vector<bool> cut1(num_graph_nodes, true);
        std::vector<bool> cut1b(num_graph_nodes, true);
        cut1b[num_graph_nodes - 1] = false;

        for (int i = 0; i < num_pixel_nodes; ++i) {
             // CORRECTED: Access the SINK enum through the full templated type
            if (g->what_segment(i) == GraphType::SINK) {
                cut1[i + 1] = false;
            }
        }
        cut1[0] = true;
        
        double sum_A_cut1b = 0;
        double sum_A_cut1 = 0;
        for (size_t i = 0; i < A_values.size(); ++i) {
            if (cut1b[A_rows[i]] && !cut1b[A_cols[i]]) sum_A_cut1b += A_values[i];
            if (cut1[A_rows[i]] && !cut1[A_cols[i]]) sum_A_cut1 += A_values[i];
        }

         
        MatrixXd cur_indST = cur_ind; 
        if (sum_A_cut1b > sum_A_cut1) {
            MatrixXb cut_matrix(sx, sy);
            for (int i = 0; i < num_pixel_nodes; ++i) {
                 // CORRECTED: Access the SINK enum through the full templated type
                 cut_matrix(i % sx, i / sx) = (g->what_segment(i) == GraphType::SINK);
            }
            cur_indST = cur_ind + cur_step.cwiseProduct(cut_matrix.cast<double>());
            //write_debug_data(iter_prefix + "_cur_ind.txt", cur_ind);
            //write_debug_data_bool(iter_prefix + "_cut_matrix.txt", cut_matrix);
        }

        //write_debug_data_vec(iter_prefix + "_cut1.txt", cut1);
        //write_debug_data_vec(iter_prefix + "_cut1b.txt", cut1b);
        //write_debug_data(iter_prefix + "_cur_indST.txt", cur_indST);
        delete g;

        cur_ind = cur_indST;
        cur_ind = cur_ind.cwiseMax(1.0).cwiseMin(static_cast<double>(num_fms));
        
        for(int i=0; i<sx; ++i) for(int j=0; j<sy; ++j) {
            //printf("index loop (%d,%d)", i, j);
            int index = static_cast<int>(round(cur_ind(i, j))) - 1;
            if (index >= 0 && index < fms.size()) fm(i, j) = fms(index);
        }

        //write_debug_data(iter_prefix + "_fm.txt", fm);

    }

    *fm_out = new double[sx * sy];
    memcpy(*fm_out, fm.data(), sx * sy * sizeof(double));
    *masksignal_out = new bool[sx * sy];
    memcpy(*masksignal_out, masksignal.data(), sx * sy * sizeof(bool));
    
    delete[] masksignal_data;
    delete[] numMinimaPerVoxel_data;
}

extern "C" {
    void graphCutIterations_c_wrapper(
        int sx, int sy, int num_acqs, int nTE,
        const std::complex<double>* images_in,
        const double* TE_in,
        double fieldStrength,
        const double* fat_frequencies_in, int num_fat_frequencies,
        double lambda_val,
        const double* range_fm_in,
        int num_fms,
        int num_iters,
        int size_clique,
        const double* residual_data,
        const double* lmap_data,
        const double* cur_ind_data,
        double** fm_out,
        bool** masksignal_out,
        const char* output_dir) {
        
        std::vector<std::complex<double>> images(images_in, images_in + (sx * sy * nTE * num_acqs));
        std::vector<double> TE(TE_in, TE_in + nTE);
        std::vector<double> fat_frequencies(fat_frequencies_in, fat_frequencies_in + num_fat_frequencies);
        std::vector<double> range_fm(range_fm_in, range_fm_in + 2);

        graphCutIterations_cpp(
            sx, sy, num_acqs, nTE, images, TE, fieldStrength,
            fat_frequencies, lambda_val, range_fm, num_fms, num_iters, size_clique,
            residual_data, lmap_data, cur_ind_data,
            fm_out, masksignal_out, output_dir
        );
    }

    void free_memory_cpp(void* ptr) {
        delete[] static_cast<char*>(ptr);
    }
}
