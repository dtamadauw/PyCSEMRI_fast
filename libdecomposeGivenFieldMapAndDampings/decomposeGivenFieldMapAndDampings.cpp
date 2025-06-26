#include "decomposeGivenFieldMapAndDampings.h"
#include <Eigen/Dense>
#include <vector>
#include <complex>
#include <cmath>

// Portable definition of PI
const double PI = 3.14159265358979323846;

/**
 * @brief Helper struct to map separate real and imaginary C-style arrays to an
 * Eigen-like complex matrix representation without copying the data. This version
 * is templated on the memory layout (Options) to be flexible.
 */
template<int Rows, int Cols, int Options>
struct ComplexMapView {
    Eigen::Map<Eigen::Matrix<double, Rows, Cols, Options>> real_part;
    Eigen::Map<Eigen::Matrix<double, Rows, Cols, Options>> imag_part;

    ComplexMapView(double* r_ptr, double* i_ptr, Eigen::Index rows, Eigen::Index cols)
        : real_part(r_ptr, rows, cols), imag_part(i_ptr, rows, cols) {}

    // Assigns an Eigen complex matrix to the separate real/imag pointers.
    ComplexMapView& operator=(const Eigen::Matrix<std::complex<double>, Rows, Cols, Options>& matrix) {
        real_part = matrix.real();
        imag_part = matrix.imag();
        return *this;
    }
};


void decomposeGivenFieldMapAndDampings_cpp(
    const double* images_real, const double* images_imag, const double* fieldmap,
    const double* r2starWater, const double* r2starFat, const double* t_ptr,
    const double* deltaF_ptr, const double* relAmps_ptr, int sx, int sy, int C,
    int N, int num_relAmps, double ampW, int precessionIsClockwise,
    double* amps_out_real, double* amps_out_imag, int debug_kx, int debug_ky,
    double* B1_out_real, double* B1_out_imag, double* B_out_real,
    double* B_out_imag, double* s_out_real, double* s_out_imag) {

    // Per user specification, C (coils) is always 1.
    const int total_pixels = sx * sy;
    const int image_size = total_pixels * N;

    // Combine real and imaginary parts into a single complex vector for processing.
    std::vector<std::complex<double>> images_vec(image_size);
    for (int i = 0; i < image_size; ++i) {
        images_vec[i] = std::complex<double>(images_real[i], images_imag[i]);
    }
    
    // Conjugate image data if precession is not clockwise.
    if (precessionIsClockwise <= 0) {
        for (auto& val : images_vec) {
            val = std::conj(val);
        }
    }

    // Map input C-style arrays to Eigen types for efficient computation.
    Eigen::Map<const Eigen::VectorXd> t(t_ptr, N);
    Eigen::Map<const Eigen::VectorXd> deltaF(deltaF_ptr, num_relAmps + 1);
    Eigen::Map<const Eigen::VectorXd> relAmps(relAmps_ptr, num_relAmps);

    // Pre-compute the B1 matrix which is constant across all pixels.
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 2, Eigen::RowMajor> B1(N, 2);
    const std::complex<double> I(0.0, 1.0);

    for (int n = 0; n < N; ++n) {
        B1(n, 0) = ampW * std::exp(I * 2.0 * PI * deltaF(0) * t(n));
        
        std::complex<double> fat_sum = 0.0;
        for (int i = 0; i < num_relAmps; ++i) {
            fat_sum += relAmps(i) * std::exp(I * 2.0 * PI * deltaF(i + 1) * t(n));
        }
        B1(n, 1) = fat_sum;
    }

    // If debugging, save the B1 matrix.
    const bool is_debug_run = (debug_kx >= 0 && debug_ky >= 0);
    if (is_debug_run) {
        ComplexMapView<Eigen::Dynamic, 2, Eigen::RowMajor> B1_out(B1_out_real, B1_out_imag, N, 2);
        B1_out = B1;
    }

    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 2, Eigen::RowMajor> B(N, 2);
    Eigen::VectorXcd s(N); // This is a column vector by Eigen's convention.

    // Loop over each pixel to solve the least-squares problem.
    for (int kx = 0; kx < sx; ++kx) {
        for (int ky = 0; ky < sy; ++ky) {
            const int pixel_1d_idx = kx * sy + ky;
            const double fm_val = fieldmap[pixel_1d_idx];
            const double r2sW_val = r2starWater[pixel_1d_idx];
            const double r2sF_val = r2starFat[pixel_1d_idx];

            // Construct the B matrix for the current pixel.
            for (int n = 0; n < N; ++n) {
                std::complex<double> common_term = std::exp(I * 2.0 * PI * fm_val * t(n));
                B(n, 0) = B1(n, 0) * common_term * std::exp(-r2sW_val * t(n));
                B(n, 1) = B1(n, 1) * common_term * std::exp(-r2sF_val * t(n));
            }
            
            // Extract the signal vector 's' for the current pixel.
            for (int n = 0; n < N; ++n) {
                // Image layout is flattened (sx, sy, N)
                s(n) = images_vec[pixel_1d_idx * N + n];
            }
            
            // Solve the linear system B * x = s for x (the amplitudes).
            Eigen::Vector2cd x = B.colPivHouseholderQr().solve(s);

            // Store the resulting amplitudes in the output arrays.
            // Output amps layout is (sx, sy, 2, C=1)
            const int amps_base_idx = pixel_1d_idx * 2;
            amps_out_real[amps_base_idx] = x(0).real();
            amps_out_imag[amps_base_idx] = x(0).imag();
            amps_out_real[amps_base_idx + 1] = x(1).real();
            amps_out_imag[amps_base_idx + 1] = x(1).imag();
            
            // If this is the designated debug pixel, save its intermediate B and s values.
            if (is_debug_run && kx == debug_kx && ky == debug_ky) {
                 ComplexMapView<Eigen::Dynamic, 2, Eigen::RowMajor> B_out(B_out_real, B_out_imag, N, 2);
                 B_out = B;
                 
                 // For the 's' vector, we must use ColMajor to satisfy Eigen's static assert.
                 ComplexMapView<Eigen::Dynamic, 1, Eigen::ColMajor> s_out(s_out_real, s_out_imag, N, 1);
                 s_out = s;
            }
        }
    }
}
