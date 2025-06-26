#include "findLocalMinima.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>

// Function to replicate np.roll(a, -1) along the first axis
Eigen::Tensor<bool, 1, Eigen::RowMajor> roll_negative_one(const Eigen::Tensor<bool, 1, Eigen::RowMajor>& input) {
    Eigen::Tensor<bool, 1, Eigen::RowMajor> rolled_tensor(input.dimension(0));
    for (int i = 0; i < input.dimension(0) - 1; ++i) {
        rolled_tensor(i) = input(i + 1);
    }
    rolled_tensor(input.dimension(0) - 1) = input(0);
    return rolled_tensor;
}

void findLocalMinima_cpp(
    const double* residual_data, int L, int sx, int sy, double threshold, const bool* input_mask_data,
    bool* masksignal_out_data,
    std::vector<double>& resLocalMinima_out_vec,
    long& resLocalMinima_rows,
    double* numMinimaPerVoxel_out_data,
    bool debug,
    IntermediateResults* intermediates)
{
    // Map input arrays to Eigen Tensors
    Eigen::TensorMap<const Eigen::Tensor<double, 3, Eigen::RowMajor>> residual(residual_data, L, sx, sy);

    // Get dimensions
    Eigen::array<long, 3> dims = {L, sx, sy};

    // Calculate dres = np.diff(residual, axis=0)
    Eigen::Tensor<double, 3, Eigen::RowMajor> dres(L - 1, sx, sy);
    dres = residual.slice(Eigen::array<long, 3>{1, 0, 0}, Eigen::array<long, 3>{L - 1, sx, sy}) -
           residual.slice(Eigen::array<long, 3>{0, 0, 0}, Eigen::array<long, 3>{L - 1, sx, sy});
    if (debug) {
        intermediates->dres = new double[(L - 1) * sx * sy];
        Eigen::Map<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(intermediates->dres, (L - 1), sx * sy) =
            Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(dres.data(), L-1, sx*sy);
    }

    // Initialize masksignal
    Eigen::TensorMap<Eigen::Tensor<bool, 2, Eigen::RowMajor>> masksignal_out(masksignal_out_data, sx, sy);

    if (input_mask_data == nullptr) {
        Eigen::array<int, 1> sum_axis = {0};
        Eigen::Tensor<double, 2, Eigen::RowMajor> sumres_tensor = residual.sum(sum_axis).sqrt();

        if (debug) {
            intermediates->sumres = new double[sx * sy];
            memcpy(intermediates->sumres, sumres_tensor.data(), sx * sy * sizeof(double));
        }

        Eigen::Tensor<double, 0, Eigen::RowMajor> max_sumres_tensor = sumres_tensor.maximum();
        const double max_sumres = max_sumres_tensor();

        if (max_sumres > 0) {
            sumres_tensor = sumres_tensor / max_sumres;
        }
        masksignal_out = sumres_tensor > threshold;

        if (debug) {
            intermediates->initial_masksignal = new bool[sx * sy];
            memcpy(intermediates->initial_masksignal, masksignal_out.data(), sx * sy * sizeof(bool));
        }
    } else {
        Eigen::TensorMap<const Eigen::Tensor<bool, 2, Eigen::RowMajor>> input_mask(input_mask_data, sx, sy);
        masksignal_out = input_mask;
    }

    // Initialize outputs
    Eigen::Tensor<double, 3, Eigen::RowMajor> resLocalMinima(1, sx, sy);
    resLocalMinima.setZero();
    Eigen::TensorMap<Eigen::Tensor<double, 2, Eigen::RowMajor>> numMinimaPerVoxel(numMinimaPerVoxel_out_data, sx, sy);
    numMinimaPerVoxel.setZero();

    if (debug) {
        intermediates->temp_steps = new std::vector<double*>();
        intermediates->temp_kx = new std::vector<int>();
        intermediates->temp_ky = new std::vector<int>();
    }

    // Main loop
    for (int kx = 0; kx < sx; ++kx) {
        for (int ky = 0; ky < sy; ++ky) {
            if (masksignal_out(kx, ky)) {
                auto voxel_residual_slice = residual.chip(kx, 1).chip(ky, 1);

                Eigen::Tensor<double, 0, Eigen::RowMajor> minres_tensor = voxel_residual_slice.minimum();
                double minres = minres_tensor();

                Eigen::Tensor<double, 0, Eigen::RowMajor> maxres_tensor = voxel_residual_slice.maximum();
                double maxres = maxres_tensor();


                Eigen::Tensor<double, 1, Eigen::RowMajor> dres_slice(L-1);
                dres_slice = dres.chip(kx,1).chip(ky,1);

                Eigen::Tensor<double, 1, Eigen::RowMajor> temp_dres_padded(L);
                temp_dres_padded.setZero();
                temp_dres_padded.slice(Eigen::array<long, 1>{1}, Eigen::array<long, 1>{L - 1}) = dres_slice;

                Eigen::Tensor<bool, 1, Eigen::RowMajor> temp_logical_part1 = temp_dres_padded < 0.0;
                Eigen::Tensor<bool, 1, Eigen::RowMajor> temp_logical_part2 = roll_negative_one((temp_dres_padded > 0.0));
                Eigen::Tensor<bool, 1, Eigen::RowMajor> temp_logical_part3 = voxel_residual_slice < (minres + 0.3 * (maxres - minres));

                Eigen::Tensor<bool, 1, Eigen::RowMajor> temp = temp_logical_part1 && temp_logical_part2 && temp_logical_part3;

                Eigen::Tensor<long, 0, Eigen::RowMajor> num_minima_tensor = temp.cast<long>().sum();
                long num_minima_here = num_minima_tensor();

                if (num_minima_here > 0) {
                    if (debug) {
                        double* temp_dump = new double[L];
                        for(int i=0; i<L; ++i) temp_dump[i] = temp(i);
                        intermediates->temp_steps->push_back(temp_dump);
                        intermediates->temp_kx->push_back(kx);
                        intermediates->temp_ky->push_back(ky);
                    }

                    if (num_minima_here > resLocalMinima.dimension(0)) {
                        long rows_to_add = num_minima_here - resLocalMinima.dimension(0);
                        Eigen::Tensor<double, 3, Eigen::RowMajor> new_res(num_minima_here, sx, sy);
                        new_res.setZero();

                        Eigen::array<long, 3> offsets = {rows_to_add, 0, 0};
                        Eigen::array<long, 3> extents = {resLocalMinima.dimension(0), sx, sy};
                        new_res.slice(offsets, extents) = resLocalMinima;
                        resLocalMinima = new_res;
                    }

                    std::vector<int> indices;
                    for (int i = 0; i < L; ++i) {
                        if (temp(i)) {
                            indices.push_back(i);
                        }
                    }

                    // Clear the column before filling
                    resLocalMinima.chip(ky, 2).chip(kx, 1).setZero();

                    for (size_t i = 0; i < indices.size(); ++i) {
                        resLocalMinima(i, kx, ky) = indices[i];
                    }
                }
                numMinimaPerVoxel(kx, ky) = num_minima_here;
            }
        }
    }

    resLocalMinima_rows = resLocalMinima.dimension(0);
    resLocalMinima_out_vec.assign(resLocalMinima.data(), resLocalMinima.data() + resLocalMinima.size());
}


extern "C" {
    void findLocalMinima_cpp_debug(
        const double* residual, int L, int sx, int sy,
        double threshold,
        const bool* masksignal_in,
        bool* masksignal_out,
        double** resLocalMinima_out,
        long* resLocalMinima_dims,
        double* numMinimaPerVoxel_out,
        IntermediateResults* intermediates)
    {
        std::vector<double> resLocalMinima_vec;
        long res_rows = 0;

        findLocalMinima_cpp(
            residual, L, sx, sy, threshold, masksignal_in,
            masksignal_out, resLocalMinima_vec, res_rows, numMinimaPerVoxel_out,
            false, intermediates
        );

        *resLocalMinima_out = new double[res_rows * sx * sy];
        memcpy(*resLocalMinima_out, resLocalMinima_vec.data(), res_rows * sx * sy * sizeof(double));

        resLocalMinima_dims[0] = res_rows;
        resLocalMinima_dims[1] = sx;
        resLocalMinima_dims[2] = sy;
    }

    void free_memory(double* ptr) {
        delete[] ptr;
    }
    void free_bool_memory(bool* ptr) {
        delete[] ptr;
    }
    void free_intermediates(IntermediateResults* intermediates) {
        if (!intermediates) return;
        if(intermediates->dres) delete[] intermediates->dres;
        if(intermediates->initial_maxres) delete[] intermediates->initial_maxres;
        if(intermediates->initial_minres) delete[] intermediates->initial_minres;
        if(intermediates->sumres) delete[] intermediates->sumres;
        if(intermediates->initial_masksignal) delete[] intermediates->initial_masksignal;
        if (intermediates->temp_steps) {
            for (auto p : *intermediates->temp_steps) {
                delete[] p;
            }
            delete intermediates->temp_steps;
        }
        if(intermediates->temp_kx) delete intermediates->temp_kx;
        if(intermediates->temp_ky) delete intermediates->temp_ky;
    }
}