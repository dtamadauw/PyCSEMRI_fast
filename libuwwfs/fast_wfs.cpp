#include "fast_wfs.h"
#include <numeric> // For std::iota, std::accumulate
#include <algorithm> // For std::sort, std::min_element, etc.
#include <limits> // For std::numeric_limits
#include <string> // For std::to_string

// Helper function to write an Eigen vector to a text file for debugging
void write_vector_to_file(const Eigen::VectorXd& vec, const std::string& filename) {
#ifdef FAST_WFS_DEBUG
    std::ofstream file(filename);
    if (file.is_open()) {
        file << std::fixed << std::setprecision(8);
        for (long long i = 0; i < vec.size(); ++i) {
            file << vec(i) << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file for writing: " << filename << std::endl;
    }
#endif
}

// Helper function to write an Eigen complex matrix (real/imag separate)
void write_complex_matrix_to_file(const Eigen::MatrixXcd& mat, const std::string& filename_prefix) {
#ifdef FAST_WFS_DEBUG
    // .real() and .imag() return expression templates. Evaluate them into a
    // concrete matrix to get a valid .data() pointer.
    Eigen::MatrixXd mat_real = mat.real();
    Eigen::MatrixXd mat_imag = mat.imag();
    write_vector_to_file(Eigen::Map<const Eigen::VectorXd>(mat_real.data(), mat.size()), filename_prefix + "_real.txt");
    write_vector_to_file(Eigen::Map<const Eigen::VectorXd>(mat_imag.data(), mat.size()), filename_prefix + "_imag.txt");
#endif
}

// --- NEW DEBUG HELPER for complex vector ---
void write_complex_vector_to_file(const Eigen::VectorXcd& vec, const std::string& filename_prefix) {
#ifdef FAST_WFS_DEBUG
    Eigen::MatrixXd vec_real = vec.real();
    Eigen::MatrixXd vec_imag = vec.imag();
    write_vector_to_file(Eigen::Map<const Eigen::VectorXd>(vec_real.data(), vec_real.size()), filename_prefix + "_real.txt");
    write_vector_to_file(Eigen::Map<const Eigen::VectorXd>(vec_imag.data(), vec_imag.size()), filename_prefix + "_imag.txt");
#endif
}


// --- Statistical Helper Functions (for fm range finding) ---

// Calculates the median of a std::vector
double median(std::vector<double>& v) {
    if (v.empty()) return 0.0;
    size_t n = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + n, v.end());
    double med = v[n];
    if (v.size() % 2 == 0) {
        std::nth_element(v.begin(), v.begin() + n - 1, v.end());
        med = (med + v[n - 1]) / 2.0;
    }
    return med;
}

// Calculates a percentile of a std::vector
double percentile(std::vector<double>& v, double p) {
    if (v.empty()) return 0.0;
    // Ensure p is in [0, 1]
    p = std::max(0.0, std::min(1.0, p));
    if (v.size() == 1) return v[0];

    double p_idx = p * (v.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(p_idx));
    size_t hi = static_cast<size_t>(std::ceil(p_idx));
    double frac = p_idx - lo;
    
    // Nth_element is faster than full sort
    std::nth_element(v.begin(), v.begin() + lo, v.end());
    double val_lo = v[lo];
    
    if (frac == 0.0) return val_lo; // Exact index

    std::nth_element(v.begin(), v.begin() + hi, v.end());
    double val_hi = v[hi];
    
    return val_lo + (val_hi - val_lo) * frac;
}

// Calculates the standard deviation of a std::vector
double std_dev(const std::vector<double>& v) {
    if (v.size() < 2) return 0.0;

    // Calculate mean
    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();

    // Calculate variance
    double sq_sum = 0.0;
    for(const auto& val : v) {
        sq_sum += (val - mean) * (val - mean);
    }
    double var = sq_sum / (v.size() - 1); // Sample std dev

    return std::sqrt(var);
}

// --- End Statistical Helpers ---


// --- OPTIMIZED Median Filter (Strategy 5b) ---
Eigen::VectorXd median_filter_3x3(const Eigen::VectorXd& in_map, int nx, int ny) {
    Eigen::VectorXd out_map = in_map;
    // Use fixed-size array and nth_element (much faster than vector+sort)
    std::array<double, 9> window;

    for (int c = 0; c < ny; ++c) {
        for (int r = 0; r < nx; ++r) {
            int w_idx = 0;
            for (int dc = -1; dc <= 1; ++dc) {
                for (int dr = -1; dr <= 1; ++dr) {
                    int nr = r + dr;
                    int nc = c + dc;
                    
                    // Symmetric padding (clamp to edge)
                    nr = std::max(0, std::min(nx - 1, nr));
                    nc = std::max(0, std::min(ny - 1, nc));
                    
                    long long n_idx = nr + (long long)nc * nx;
                    window[w_idx++] = in_map(n_idx);
                }
            }

            // Find the median (element 4) of the 9 elements
            std::nth_element(window.begin(), window.begin() + 4, window.end());
            out_map(r + (long long)c * nx) = window[4];
        }
    }
    return out_map;
}

// --- NEW: Mask-aware separable box blur (approximates Gaussian) ---

// --- COMPILER FIX: Templated to use Eigen::DenseBase ---
template<typename SrcMatrix, typename DstMatrix>
static void smooth_pass_horizontal_inplace(
    const Eigen::DenseBase<SrcMatrix>& src_map_base, // Read from
    Eigen::DenseBase<DstMatrix>& dst_map_base,       // Write to
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius)
{
    // Get the actual derived matrix from the base
    const auto& src_map = src_map_base.derived();
    auto& dst_map = dst_map_base.derived();

    int nx = src_map.rows();
    int ny = src_map.cols();
    
    for (int c = 0; c < ny; ++c) {
        for (int r = 0; r < nx; ++r) {
            if (mask(r, c) < 0.1) {
                dst_map(r, c) = src_map(r, c); // Copy unmasked values
                continue;
            }
            double sum = 0.0;
            double weight = 0.0;
            for (int i = -radius; i <= radius; ++i) {
                int rr = std::max(0, std::min(nx - 1, r + i)); // Clamp edge
                if (mask(rr, c) > 0.1) {
                    sum += src_map(rr, c); // Read from source map
                    weight += 1.0;
                }
            }
            if (weight > 0.0) {
                dst_map(r, c) = sum / weight; // Write to destination map
            } else {
                dst_map(r, c) = src_map(r, c); // Fallback if no neighbors
            }
        }
    }
}

// --- COMPILER FIX: Templated to use Eigen::DenseBase ---
template<typename SrcMatrix, typename DstMatrix>
static void smooth_pass_vertical_inplace(
    const Eigen::DenseBase<SrcMatrix>& src_map_base, // Read from
    Eigen::DenseBase<DstMatrix>& dst_map_base,       // Write to
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius)
{
    // Get the actual derived matrix from the base
    const auto& src_map = src_map_base.derived();
    auto& dst_map = dst_map_base.derived();

    int nx = src_map.rows();
    int ny = src_map.cols();

    for (int r = 0; r < nx; ++r) {
        for (int c = 0; c < ny; ++c) {
            if (mask(r, c) < 0.1) {
                dst_map(r, c) = src_map(r, c); // Copy unmasked values
                continue;
            }
            double sum = 0.0;
            double weight = 0.0;
            for (int j = -radius; j <= radius; ++j) {
                int cc = std::max(0, std::min(ny - 1, c + j)); // Clamp edge
                if (mask(r, cc) > 0.1) {
                    sum += src_map(r, cc); // Read from source map
                    weight += 1.0;
                }
            }
            if (weight > 0.0) {
                dst_map(r, c) = sum / weight; // Write to destination map
            } else {
                dst_map(r, c) = src_map(r, c); // Fallback if no neighbors
            }
        }
    }
}


// Main smoothing function
// --- MODIFICATION: More efficient iteration ---
static void smooth_with_mask_inplace(
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& map,
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>& mask,
    int radius, int passes)
{
    // Allocate one temporary map outside the loop
    Eigen::MatrixXd temp_map(map.rows(), map.cols());
    
    for (int p = 0; p < passes; ++p) {
        // Pass 1: Read from map, Write to temp_map
        smooth_pass_horizontal_inplace(map, temp_map, mask, radius);
        // Pass 2: Read from temp_map, Write back to map
        // --- COMPILER FIX: 'map' is now a valid destination ---
        smooth_pass_vertical_inplace(temp_map, map, mask, radius); 
    }
}
// --- END NEW SMOOTHING ---


// --- MODIFIED: stage5 to accept 2D Eigen::Maps ---
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
)
{
    int nx = fm_map.rows();
    int ny = fm_map.cols();
    int nTE = t_n.size();
    long long n_voxels = (long long)nx * ny;

    Eigen::MatrixXcd A(nTE, 2);
    Eigen::VectorXcd s_q(nTE);

    for (int c = 0; c < ny; ++c) {
        for (int r = 0; r < nx; ++r) {
            
            // Only process masked voxels
            if (mask(r, c) < 0.1) continue;

            double fm_q = fm_map(r, c);
            double r2_q = r2s_map(r, c);

            // --- BUG FIX: Clamp R2* to be non-negative ---
            if (r2_q < 0.0) r2_q = 0.0;
            
            // Extract the high-res signal for this voxel
            long long idx_hr = r + (long long)c * nx;
            for (int t = 0; t < nTE; ++t) {
                s_q(t) = {
                    imDataParams->images_r[idx_hr + (long long)t * n_voxels],
                    imDataParams->images_i[idx_hr + (long long)t * n_voxels]
                };
            }

            Eigen::VectorXcd P_q = (t_n.array() * std::complex<double>(0.0, 2.0 * M_PI * fm_q)).exp();
            Eigen::VectorXcd D_q = (-t_n.array() * r2_q).exp();

            A.col(0) = D_q.array() * P_q.array();
            A.col(1) = A.col(0).array() * b_fat.array();

            // Solve 2xN complex linear least-squares
            Eigen::Matrix2cd AtA = A.adjoint() * A;
            double ridge = 1e-9 * (AtA(0,0).real() + AtA(1,1).real() + std::numeric_limits<double>::epsilon());
            AtA(0,0) += ridge;
            AtA(1,1) += ridge;

            Eigen::Vector2cd x_ls = AtA.ldlt().solve(A.adjoint() * s_q);
            
            // Write directly to output maps
            wat_r_amp(r, c) = x_ls(0).real();
            wat_i_amp(r, c) = x_ls(0).imag();
            fat_r_amp(r, c) = x_ls(1).real();
            fat_i_amp(r, c) = x_ls(1).imag();
        }
    }
}


// --- REFACTORED for Coarse-to-Fine Optimization ---
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
)
{
    // --- Get coarse grid sizes ---
    int F = algoParams.NUM_FMS;
    int R = algoParams.NUM_R2STARS;
    Eigen::VectorXd f_coarse = Eigen::VectorXd::LinSpaced(F, algoParams.range_fm[0], algoParams.range_fm[1]);
    Eigen::VectorXd r2_grid = Eigen::VectorXd::LinSpaced(R, algoParams.range_r2star[0], algoParams.range_r2star[1]);
    int nTE = t_n.size();

    // --- Precompute transposed signals ---
    Eigen::MatrixXcd S_t_lr = s_q_vec_lr.transpose(); // [N x Q_lr]
    Eigen::MatrixXcd S_t_fast = s_q_vec_fast.transpose(); // [N x Q_fast]

    // --- Pre-allocate buffers for MAX size ---
    long long q_max = std::max(q_masked_lr, q_masked_fast);
    Eigen::MatrixXcd S_t_demod(nTE, q_max);
    Eigen::MatrixXd min_R_per_f(q_max, F);
    Eigen::MatrixXi bestR2idx_f(q_max, F);
    Eigen::VectorXd b_q(q_max);
    Eigen::VectorXd a_q(q_max);
    Eigen::VectorXd r2s_best_q(q_max);
    
    // --- Pre-allocate temporaries for inner loop ---
    Eigen::VectorXcd gW(nTE);
    Eigen::VectorXcd gF(nTE);
    Eigen::RowVectorXcd y1_q(q_max);
    Eigen::RowVectorXcd y2_q(q_max);
    Eigen::RowVectorXcd T1_q(q_max);
    Eigen::RowVectorXcd T2_q(q_max);
    Eigen::RowVectorXd proj_energy_q(q_max);
    Eigen::VectorXd resid_q(q_max);
    Eigen::VectorXd bestR_j = Eigen::VectorXd(q_max);
    Eigen::VectorXi bestI_j = Eigen::VectorXi(q_max);

    std::cout << "  Starting iterative field map range determination..." << std::endl;
    double center_offset = 0.0; // This accumulates the offset
    
    
    // --- Define a helper lambda for the main candidate search ---
    // This runs the full F/R loop and returns b_q (best fm)
    auto run_candidate_search = [&](
        Eigen::VectorXd& b_q_out, 
        Eigen::VectorXd& a_q_out, 
        Eigen::VectorXd& r2s_best_q_out,
        const Eigen::MatrixXcd& S_t_in, // Transposed signal [N x Q]
        long long q_masked_in,
        double current_offset
    )
    {
        // 1. Demodulate signal
        if (current_offset == 0.0) {
            S_t_demod.leftCols(q_masked_in) = S_t_in;
        } else {
            Eigen::VectorXcd offset_freq_vec = (t_n.array() * std::complex<double>(0.0, -2.0 * M_PI * current_offset)).exp();
            S_t_demod.leftCols(q_masked_in).noalias() = (S_t_in.array().colwise() * offset_freq_vec.array()).matrix();
        }
        
        // 2. Get signal energy
        Eigen::VectorXd sq_norm2 = S_t_demod.leftCols(q_masked_in).colwise().squaredNorm().transpose();

        // 3. Reset minima maps
        min_R_per_f.block(0, 0, q_masked_in, F).setConstant(std::numeric_limits<double>::infinity());
        bestR2idx_f.block(0, 0, q_masked_in, F).setZero();

        // 4. Run F/R loop
        for (int j = 0; j < F; ++j) {
            bestR_j.head(q_masked_in).setConstant(std::numeric_limits<double>::infinity());
            bestI_j.head(q_masked_in).setZero();
            
            const auto& P_j = P_coarse.col(j);
            const auto& Pbf_j = Pbf.col(j);
            
            for (int i = 0; i < R; ++i) {
                const auto& D_i = D_r2_lut.col(i);
                gW.noalias() = (D_i.array() * P_j.array()).matrix();
                gF.noalias() = (D_i.array() * Pbf_j.array()).matrix();

                const auto& D2_i = D2_lut.col(i);
                double gww = D2_i.sum();
                double gff = (D2_i.array() * absB2.array()).sum();
                std::complex<double> gwf = (D2_i.array() * b_fat.array()).sum();

                double detG = gww * gff - std::norm(gwf);
                const double eps = std::numeric_limits<double>::epsilon();
                if (std::abs(detG) < 1e-14 * (std::abs(gww) + std::abs(gff) + eps)) {
                    double ridge = 1e-12 * (std::abs(gww) + std::abs(gff) + 1.0) + eps;
                    gww += ridge;
                    gff += ridge;
                    detG = gww * gff - std::norm(gwf);
                }
                double detG_inv = 1.0 / detG;
                double invG11_real = gff * detG_inv;
                double invG22_real = gww * detG_inv;
                std::complex<double> invG12 = -gwf * detG_inv;
                std::complex<double> invG21 = -std::conj(gwf) * detG_inv;
                
                y1_q.head(q_masked_in).noalias() = gW.adjoint() * S_t_demod.leftCols(q_masked_in);
                y2_q.head(q_masked_in).noalias() = gF.adjoint() * S_t_demod.leftCols(q_masked_in);
                
                T1_q.head(q_masked_in).noalias() = (invG11_real * y1_q.head(q_masked_in).array() + invG12 * y2_q.head(q_masked_in).array()).matrix();
                T2_q.head(q_masked_in).noalias() = (invG21 * y1_q.head(q_masked_in).array() + invG22_real * y2_q.head(q_masked_in).array()).matrix();
                proj_energy_q.head(q_masked_in).noalias() = ( (y1_q.head(q_masked_in).conjugate().array() * T1_q.head(q_masked_in).array() + y2_q.head(q_masked_in).conjugate().array() * T2_q.head(q_masked_in).array()).real() ).matrix();
                resid_q.head(q_masked_in).noalias() = sq_norm2 - proj_energy_q.head(q_masked_in).transpose();
                resid_q.head(q_masked_in) = resid_q.head(q_masked_in).cwiseMax(0.0);

                // --- R2* BUG FIX: Force evaluation of 'm' into a concrete array ---
                const Eigen::Array<bool, Eigen::Dynamic, 1> m = (resid_q.head(q_masked_in).array() < bestR_j.head(q_masked_in).array()).eval();
                // --- END R2* BUG FIX ---
            
                bestR_j.head(q_masked_in) = m.select(resid_q.head(q_masked_in), bestR_j.head(q_masked_in).eval());
                
                // Now this loop will work because 'm' is stable
                for (long long q = 0; q < q_masked_in; ++q) {
                    if (m(q)) { 
                        bestI_j(q) = i; 
                    }
                }
            }
            min_R_per_f.col(j).head(q_masked_in) = bestR_j.head(q_masked_in);
            bestR2idx_f.col(j).head(q_masked_in) = bestI_j.head(q_masked_in);
        }

        // 5. Find best f, r2, and residual
        for (long long q = 0; q < q_masked_in; ++q) {
            Eigen::Index best_f_idx;
            min_R_per_f.row(q).head(F).minCoeff(&best_f_idx);
            
            b_q_out(q) = f_coarse(best_f_idx);
            double min_resid = min_R_per_f(q, best_f_idx);
            if (std::isfinite(min_resid)) {
                a_q_out(q) = 1.0 / (min_resid + 1e-9);
            } else {
                a_q_out(q) = 1e-9;
            }
            r2s_best_q_out(q) = r2_grid(bestR2idx_f(q, best_f_idx));
        }
    };
    
    // --- Define a helper lambda for the statistical analysis ---
    auto find_new_offset = [&](
        const Eigen::VectorXd& b_q_in, 
        long long q_masked_in
    ) -> double
    {
        double rmin = algoParams.range_fm[0];
        double rmax = algoParams.range_fm[1];
        std::vector<double> temp_v;
        temp_v.reserve(q_masked_in);
        for(long long q=0; q < q_masked_in; ++q) {
            double f = b_q_in(q);
            if (f > rmin * 0.98 && f < rmax * 0.98) {
                temp_v.push_back(f);
            }
        }
        if(temp_v.empty()) {
             std::cout << "  WARNING: No reliable voxels found. Halting iteration." << std::endl;
             return 0.0; // No offset
        }

        double hist_peak = median(temp_v);
        double stdh = 1.0 * std_dev(temp_v);
        std::vector<double> fm_reliable;
        fm_reliable.reserve(q_masked_in);
        for(long long q=0; q < q_masked_in; ++q) {
            double f = b_q_in(q);
            if (f > (hist_peak - stdh) && f < (hist_peak + stdh) && f != 0.0) {
                fm_reliable.push_back(f);
            }
        }
        if(fm_reliable.empty()) {
             std::cout << "  WARNING: No reliable voxels in std dev range. Halting iteration." << std::endl;
             return 0.0; // No offset
        }

        double peak_value = median(fm_reliable);
        std::vector<double> left_lobe, right_lobe;
        for(const auto& f : fm_reliable) {
            if(f <= peak_value) left_lobe.push_back(f);
            if(f >= peak_value) right_lobe.push_back(f);
        }
        double left_threshold = (left_lobe.empty()) ? peak_value : percentile(left_lobe, 0.10);
        double right_threshold = (right_lobe.empty()) ? peak_value : percentile(right_lobe, 0.90);
        
        return (left_threshold + right_threshold) / 2.0;
    };


    // --- *** MAIN ITERATION LOGIC (Coarse-to-Fine) *** ---

    // --- Iter 0 (FAST) ---
    std::cout << "  Iter 1 (Fast, " << (imDataParams.im_dim[0] / nx_fast) << "x)..." << std::endl;
    run_candidate_search(
        b_q, a_q, r2s_best_q, // Ouputs (resized to q_max, only head(q_masked_fast) is filled)
        S_t_fast, q_masked_fast, 0.0
    );
    double new_offset = find_new_offset(b_q.head(q_masked_fast), q_masked_fast);
    center_offset += new_offset;

    // --- Iter 1 (FAST) ---
    std::cout << "  Iter 2 (Fast, " << (imDataParams.im_dim[0] / nx_fast) << "x)..." << std::endl;
    std::cout << "    New total center = " << center_offset << std::endl;
    run_candidate_search(
        b_q, a_q, r2s_best_q, 
        S_t_fast, q_masked_fast, center_offset
    );
    new_offset = find_new_offset(b_q.head(q_masked_fast), q_masked_fast);
    center_offset += new_offset;

    // --- Iter 2 (PLANNED) ---
    std::cout << "  Iter 3 (Planned, " << algoParams.SUBSAMPLE << "x)..." << std::endl;
    std::cout << "    New total center = " << center_offset << std::endl;
    run_candidate_search(
        b_q, a_q, r2s_best_q, // Final candidates
        S_t_lr, q_masked_lr, center_offset
    );
    
    // --- *** END OF ITERATION LOGIC *** ---


    std::cout << "  ...Iterative search complete. Final offset: " << center_offset << std::endl;
#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_b_q_pre_filter.txt" << std::endl;
    write_vector_to_file(b_q.head(q_masked_lr), "dbg_c_b_q_pre_filter.txt");
#endif

    // --- START: MEDIAN FILTER STEP (on 'lr' data) ---
    std::cout << "  Applying 3x3 median filter to initial field map..." << std::endl;
    Eigen::VectorXd fm_init_noisy_lr(nx_lr * ny_lr);
    fm_init_noisy_lr.setZero();
    long long current_q_pack = 0;
    for(long long i=0; i < (nx_lr * ny_lr); ++i) {
        if(mask_vec_lr(i) > 0) {
            if (current_q_pack < q_masked_lr) {
                fm_init_noisy_lr(i) = b_q(current_q_pack);
                current_q_pack++;
            }
        }
    }
    Eigen::VectorXd fm_init_filtered_lr = median_filter_3x3(fm_init_noisy_lr, nx_lr, ny_lr);
    
    // 3. Pack the filtered 2D map back into the 'b_q' vector
    //    AND add the final center offset
    current_q_pack = 0;
    for(long long i=0; i < (nx_lr * ny_lr); ++i) {
        if(mask_vec_lr(i) > 0) {
             if (current_q_pack < q_masked_lr) {
                b_q(current_q_pack) = fm_init_filtered_lr(i) + center_offset;
                current_q_pack++;
            }
        }
    }
    std::cout << "  ...Median filter complete." << std::endl;
#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_b_q_post_filter.txt" << std::endl;
    write_vector_to_file(b_q.head(q_masked_lr), "dbg_c_b_q_post_filter.txt");
#endif
    
    // --- Stage 3: Spatial Coupling (on 'lr' data) ---
    std::cout << "  Starting Stage 3: Spatial Coupling..." << std::endl;
    Eigen::SparseMatrix<double> L_w(q_masked_lr, q_masked_lr);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(q_masked_lr * 9); 
    Eigen::VectorXi map_idx = Eigen::VectorXi::Zero(nx_lr * ny_lr);
    long long current_q = 0;
    for(long long i=0; i < (nx_lr * ny_lr); ++i) if(mask_vec_lr(i) > 0) map_idx(i) = current_q++;

    Eigen::VectorXd diag_sums = Eigen::VectorXd::Zero(q_masked_lr);
    for (long long i = 0; i < (nx_lr * ny_lr); ++i) {
        if (mask_vec_lr(i) == 0) continue;
        long long q_idx = map_idx(i);
        long long r = i % nx_lr;
        long long c = i / nx_lr;

        int dr_neighbors[] = {0, 1, 1, 1};
        int dc_neighbors[] = {1, -1, 0, 1};
        
        for(int k=0; k < 4; ++k) {
            long long nr = r + dr_neighbors[k];
            long long nc = c + dc_neighbors[k];
            if (nr >= 0 && nr < nx_lr && nc >= 0 && nc < ny_lr) {
                long long neighbor_i = nr + nc * nx_lr;
                if (mask_vec_lr(neighbor_i) > 0) {
                    long long j_idx = map_idx(neighbor_i);
                    // --- BUG FIX: check j_idx > 0 (is valid neighbor) and q_idx != j_idx ---
                    if (q_idx != j_idx && q_idx < q_masked_lr && j_idx < q_masked_lr && j_idx >= 0) { 
                        double weight = std::min(a_q(q_idx), a_q(j_idx));
                        tripletList.push_back(Eigen::Triplet<double>(q_idx, j_idx, -weight));
                        tripletList.push_back(Eigen::Triplet<double>(j_idx, q_idx, -weight));
                        diag_sums(q_idx) += weight;
                        diag_sums(j_idx) += weight;
                    }
                }
            }
        }
    }
    for(long long q=0; q < q_masked_lr; ++q) {
        tripletList.push_back(Eigen::Triplet<double>(q, q, diag_sums(q)));
    }

    L_w.setFromTriplets(tripletList.begin(), tripletList.end());
    
    Eigen::SparseMatrix<double> A_matrix(q_masked_lr, q_masked_lr);
    A_matrix = L_w * algoParams.mu;
    A_matrix.diagonal() += a_q.head(q_masked_lr); // Use head()
    Eigen::VectorXd rhs = a_q.head(q_masked_lr).array() * b_q.head(q_masked_lr).array();

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> solver;
    solver.setTolerance(1e-3); 
    solver.setMaxIterations(100); 
    solver.compute(A_matrix);
    fm_init = solver.solve(rhs); // Final output vector
    std::cout << "  ...Spatial Coupling complete." << std::endl;
    
    // Set the final r2s map (pass-through from Stage 2, iter 2)
    r2s_init = r2s_best_q.head(q_masked_lr);
    
#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_fm_final_lr_vec.txt" << std::endl;
    write_vector_to_file(fm_init, "dbg_c_fm_final_lr_vec.txt");
    std::cout << "  DEBUG: Wrote dbg_c_r2s_final_lr_vec.txt" << std::endl;
    write_vector_to_file(r2s_init, "dbg_c_r2s_final_lr_vec.txt");
#endif
}


// --- NEW HELPER FUNCTION for creating downsampled data ---
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
)
{
    int nx_hr = imDataParams->im_dim[0];
    int ny_hr = imDataParams->im_dim[1];
    int nTE = imDataParams->nte;
    long long n_voxels_hr = (long long)nx_hr * ny_hr;

    nx_out = (nx_hr + factor - 1) / factor; // Ceiling
    ny_out = (ny_hr + factor - 1) / factor; // Ceiling
    long long n_voxels_out = (long long)nx_out * ny_out;

    // 1. Downsample by nearest neighbor
    std::vector<double> images_r_lr(n_voxels_out * nTE);
    std::vector<double> images_i_lr(n_voxels_out * nTE);

    for (int c_lr = 0; c_lr < ny_out; ++c_lr) {
        for (int r_lr = 0; r_lr < nx_out; ++r_lr) {
            int r = r_lr * factor;
            int c = c_lr * factor;
            if (r >= nx_hr) r = nx_hr - 1;
            if (c >= ny_hr) c = ny_hr - 1;

            long long idx_hr = r + (long long)c * nx_hr;
            long long idx_lr = r_lr + (long long)c_lr * nx_out;
            for (int t = 0; t < nTE; ++t) {
                images_r_lr[idx_lr + (long long)t * n_voxels_out] = imDataParams->images_r[idx_hr + (long long)t * n_voxels_hr];
                images_i_lr[idx_lr + (long long)t * n_voxels_out] = imDataParams->images_i[idx_hr + (long long)t * n_voxels_hr];
            }
        }
    }
    
    // 2. Create Mask from this downsampled data
    mask_vec_out.resize(n_voxels_out);
    double max_mag = 0.0;
    for(long long i=0; i < n_voxels_out; ++i) {
        double mag = std::sqrt(images_r_lr[i]*images_r_lr[i] + images_i_lr[i]*images_i_lr[i]);
        if (std::isfinite(mag) && mag > max_mag) max_mag = mag;
    }
    
    q_masked_out = 0;
    for(long long i=0; i < n_voxels_out; ++i) {
        double mag = std::sqrt(images_r_lr[i]*images_r_lr[i] + images_i_lr[i]*images_i_lr[i]);
        if(mag > mask_threshold * max_mag && std::isfinite(mag)) {
            mask_vec_out(i) = 1.0;
            q_masked_out++;
        } else {
            mask_vec_out(i) = 0.0;
        }
    }
    
    // 3. Create s_q_vec from masked data
    s_q_vec_out.resize(q_masked_out, nTE);
    long long current_q = 0;
    for (long long i = 0; i < n_voxels_out; ++i) {
        if (mask_vec_out(i) > 0) {
            if (current_q < q_masked_out) { // Bounds check
                for (int j = 0; j < nTE; ++j) {
                    s_q_vec_out(current_q, j) = {images_r_lr[i + j*n_voxels_out], images_i_lr[i + j*n_voxels_out]};
                }
                current_q++;
            }
        }
    }
}


// C-style wrapper for the initial guess algorithm
void VARPRO_LUT(imDataParams_str *imDataParams, algoParams_str *algoParams, outInitParams_str* outInitParams) {
    
    std::cout << "--- Starting Fast IDEAL Reconstruction (C++) ---" << std::endl;
#ifdef FAST_WFS_DEBUG
    std::cout << "  *** DEBUG MODE ENABLED: Writing intermediate files. ***" << std::endl;
#endif
    
    // --- Full Resolution Parameters ---
    int nx = imDataParams->im_dim[0];
    int ny = imDataParams->im_dim[1];
    int nTE = imDataParams->nte;
    long long n_voxels = (long long)nx * ny;


    for (int i=0;i<n_voxels*nTE;i++){
        imDataParams->images_i[i] = -1.0*imDataParams->images_i[i];
    }

    // --- *** NEW: Coarse-to-Fine Subsampling Logic *** ---

    // 1. "Planned" LR dataset (from algoParams)
    int factor_lr = std::max(1, algoParams->SUBSAMPLE);
    Eigen::MatrixXcd s_q_vec_lr;
    Eigen::VectorXd mask_vec_lr;
    int nx_lr, ny_lr;
    long long q_masked_lr;
    
    std::cout << "Stage 0: Creating 'Planned' " << factor_lr << "x downsampled data..." << std::endl;
    create_downsampled_data(
        s_q_vec_lr, mask_vec_lr, nx_lr, ny_lr, q_masked_lr,
        imDataParams, factor_lr, algoParams->mask_threshold
    );
    std::cout << "  Masked voxels (planned): " << q_masked_lr << " / " << (nx_lr * ny_lr) << std::endl;

    // 2. "Fast" LR dataset (8x or 4x, for iterations)
    int factor_fast = 8;
    // Use ceiling division
    if ( ((nx + 7) / 8) * ((ny + 7) / 8) < 512) {
        factor_fast = 4;
    }
    // Ensure "fast" is never finer than "planned"
    factor_fast = std::max(factor_lr, factor_fast);

    Eigen::MatrixXcd s_q_vec_fast;
    Eigen::VectorXd mask_vec_fast;
    int nx_fast, ny_fast;
    long long q_masked_fast;
    
    if (factor_fast == factor_lr) {
        std::cout << "  'Fast' (" << factor_fast << "x) is same as 'Planned' (" << factor_lr << "x). Re-using data." << std::endl;
        s_q_vec_fast = s_q_vec_lr;
        mask_vec_fast = mask_vec_lr;
        nx_fast = nx_lr;
        ny_fast = ny_lr;
        q_masked_fast = q_masked_lr;
    } else {
        std::cout << "Stage 0: Creating 'Fast' " << factor_fast << "x downsampled data..." << std::endl;
        create_downsampled_data(
            s_q_vec_fast, mask_vec_fast, nx_fast, ny_fast, q_masked_fast,
            imDataParams, factor_fast, algoParams->mask_threshold
        );
        std::cout << "  Masked voxels (fast): " << q_masked_fast << " / " << (nx_fast * ny_fast) << std::endl;
    }
    
#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_mask_vec_lr.txt" << std::endl;
    write_vector_to_file(mask_vec_lr, "dbg_c_mask_vec_lr.txt");
    std::cout << "  DEBUG: Wrote dbg_c_s_q_vec_lr_real.txt" << std::endl;
    write_complex_matrix_to_file(s_q_vec_lr, "dbg_c_s_q_vec_lr");
#endif
    
    // --- END NEW SUBSAMPLING ---

    
    // --- Create High-Res Mask (for final smoothing and Stage 5) ---
    std::cout << "  Creating high-resolution mask..." << std::endl;
    Eigen::VectorXd mask_vec_hr(n_voxels);
    long long q_masked_hr = 0;
    double max_mag = 0.0;
    for(long long i=0; i < n_voxels; ++i) {
        double mag = std::sqrt(imDataParams->images_r[i]*imDataParams->images_r[i] + imDataParams->images_i[i]*imDataParams->images_i[i]);
        if (std::isfinite(mag) && mag > max_mag) max_mag = mag;
    }
    for(long long i=0; i < n_voxels; ++i) {
        double mag = std::sqrt(imDataParams->images_r[i]*imDataParams->images_r[i] + imDataParams->images_i[i]*imDataParams->images_i[i]);
        if(mag > algoParams->mask_threshold * max_mag && std::isfinite(mag)) {
            mask_vec_hr(i) = 1.0;
            q_masked_hr++;
        } else {
            mask_vec_hr(i) = 0.0;
        }
    }
    std::cout << "  Masked voxels (high-res): " << q_masked_hr << " / " << n_voxels << std::endl;
    
    Eigen::Map<Eigen::VectorXd> t_n(imDataParams->TE, nTE);

    // --- STAGE 1: Pre-compute ALL basis LUTs (Strategy 1a) ---
    std::cout << "Stage 1: Pre-calculating all basis LUTs..." << std::endl;
    
    Eigen::VectorXcd b_fat = Eigen::VectorXcd::Zero(nTE);
    for (int i = 0; i < algoParams->NUM_FAT_PEAKS; ++i) {
        for (int j = 0; j < nTE; ++j) {
            std::complex<double> i2pift = {0.0, 2.0 * M_PI * algoParams->species_fat_freq[i] * GYRO * imDataParams->FieldStrength * t_n(j)};
            b_fat(j) += algoParams->species_fat_amp[i] * exp(i2pift);
        }
    }
    
    Eigen::VectorXd absB2 = b_fat.array().abs2();

#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_b_fat_real.txt" << std::endl;
    write_complex_matrix_to_file(b_fat.transpose(), "dbg_c_b_fat"); 
#endif

    int F = algoParams->NUM_FMS;
    int R = algoParams->NUM_R2STARS;
    Eigen::VectorXd f_coarse = Eigen::VectorXd::LinSpaced(F, algoParams->range_fm[0], algoParams->range_fm[1]);
    Eigen::VectorXd r2_grid = Eigen::VectorXd::LinSpaced(R, algoParams->range_r2star[0], algoParams->range_r2star[1]);

    Eigen::MatrixXcd P_coarse(nTE, F);
    for (int j = 0; j < F; ++j) {
        P_coarse.col(j) = (t_n.array() * std::complex<double>(0.0, 2.0 * M_PI * f_coarse(j))).exp();
    }
    
    Eigen::MatrixXd D_r2_lut(nTE, R);
    for (int i = 0; i < R; ++i) {
        D_r2_lut.col(i) = (-t_n.array() * r2_grid(i)).exp();
    }
    
    Eigen::MatrixXcd Pbf = P_coarse.array().colwise() * b_fat.array();
    Eigen::MatrixXd D2_lut = D_r2_lut.array().square();

#ifdef FAST_WFS_DEBUG
    std::cout << "  DEBUG: Wrote dbg_c_f_coarse.txt" << std::endl;
    write_vector_to_file(f_coarse, "dbg_c_f_coarse.txt");
    std::cout << "  DEBUG: Wrote dbg_c_r2_grid.txt" << std::endl;
    write_vector_to_file(r2_grid, "dbg_c_r2_grid.txt");
#endif
    // --- END OF STAGE 1 ---


    // --- Call Core Algorithm (Stages 2-3) ---
    std::cout << "Stages 2-3: Candidate Generation and Spatial Coupling..." << std::endl;
    Eigen::VectorXd fm_init_lr_vec, r2s_init_lr_vec;
    
    // --- Call with new signature ---
    fast_ideal_initial_guess_cpp(
        fm_init_lr_vec, r2s_init_lr_vec, // Outputs
        s_q_vec_lr, mask_vec_lr, nx_lr, ny_lr, q_masked_lr, // Planned data
        s_q_vec_fast, mask_vec_fast, nx_fast, ny_fast, q_masked_fast, // Fast data
        t_n, *algoParams, *imDataParams, // Common
        b_fat, absB2, P_coarse, Pbf, D_r2_lut, D2_lut // LUTs
    );


    long long fm_sz=fm_init_lr_vec.size(), r2_sz=r2s_init_lr_vec.size();
    if (fm_sz != q_masked_lr || r2_sz != q_masked_lr ) {
        std::cerr << "ERROR: Size mismatch after guess! q_masked_lr=" << q_masked_lr 
                  << " fm_sz=" << fm_sz << " r2_sz=" << r2_sz << std::endl;
        q_masked_lr = std::min({fm_sz, r2_sz, q_masked_lr});
        std::cerr << "  Corrected q_masked_lr=" << q_masked_lr << std::endl;
    }

    
    // --- Unpack FM/R2S Results into Low-Res Maps ---
    std::cout << "  Unpacking fm/r2s results into low-res maps..." << std::endl;
    // --- COMPILER FIX: Use (nx_lr * ny_lr) instead of n_voxels_lr ---
    long long n_voxels_lr_calc = (long long)nx_lr * ny_lr;
    Eigen::VectorXd fm_init_lr(n_voxels_lr_calc); fm_init_lr.setZero();
    Eigen::VectorXd r2s_init_lr(n_voxels_lr_calc); r2s_init_lr.setZero();

    long long current_q_unpack = 0;
    for(long long i=0; i < n_voxels_lr_calc; ++i) {
        if(mask_vec_lr(i) > 0 && current_q_unpack < q_masked_lr) {
            fm_init_lr(i) = fm_init_lr_vec(current_q_unpack);
            r2s_init_lr(i) = r2s_init_lr_vec(current_q_unpack);
            current_q_unpack++;
        }
    }

    // --- Bilinear Interpolation Helper (for smooth fm/r2s) ---
    auto bilinear_interp = [&](const Eigen::VectorXd& low_res_data, int r_hr, int c_hr, int factor_in, int nx_lr_in, int ny_lr_in) -> double {
        double r_lr_f = (double)(r_hr + 0.5) / factor_in - 0.5;
        double c_lr_f = (double)(c_hr + 0.5) / factor_in - 0.5;
        int r0 = static_cast<int>(floor(r_lr_f));
        int c0 = static_cast<int>(floor(c_lr_f));
        int r1 = r0 + 1;
        int c1 = c0 + 1;
        // Clamp coordinates
        r0 = std::max(0, std::min(nx_lr_in - 1, r0));
        c0 = std::max(0, std::min(ny_lr_in - 1, c0));
        r1 = std::max(0, std::min(nx_lr_in - 1, r1));
        c1 = std::max(0, std::min(ny_lr_in - 1, c1));
        
        double dr = r_lr_f - r0;
        double dc = c_lr_f - c0;
        
        double v00 = low_res_data(r0 + (long long)c0 * nx_lr_in);
        double v10 = low_res_data(r1 + (long long)c0 * nx_lr_in);
        double v01 = low_res_data(r0 + (long long)c1 * nx_lr_in);
        double v11 = low_res_data(r1 + (long long)c1 * nx_lr_in);
        
        double top_interp = v00 * (1.0 - dc) + v01 * dc;
        double bottom_interp = v10 * (1.0 - dc) + v11 * dc;
        double final_val = top_interp * (1.0 - dr) + bottom_interp * dr;
        
        return final_val;
    };

    double* out_wr = outInitParams->wat_r_amp;
    double* out_wi = outInitParams->wat_i_amp; 
    double* out_fr = outInitParams->fat_r_amp;
    double* out_fi = outInitParams->fat_i_amp;

    // --- Upsampling Loop (FM/R2S only) ---
    std::cout << "  Upsampling fm/r2s results to original resolution..." << std::endl;
    if (!outInitParams || !outInitParams->fm_init || !outInitParams->r2s_init || !outInitParams->masksignal_init || !out_wr || !out_wi || !out_fr || !out_fi) {
        std::cerr << "ERROR: Output pointers NULL before final upsampling!" << std::endl; return;
    }
    
    for (int c = 0; c < ny; ++c) {
        for (int r = 0; r < nx; ++r) {
            long long idx_hr = r + (long long)c * nx;

            // Use the HIGH-RES mask
            double mask_val = mask_vec_hr(idx_hr);
            outInitParams->masksignal_init[idx_hr] = mask_val;

            if (mask_val > 0.1) {
                // --- Bilinear for FM (as requested for smoothness) ---
                outInitParams->fm_init[idx_hr] = bilinear_interp(fm_init_lr, r, c, factor_lr, nx_lr, ny_lr);
                
                // --- Bilinear for R2S (for smooth input to solver) ---
                outInitParams->r2s_init[idx_hr] = bilinear_interp(r2s_init_lr, r, c, factor_lr, nx_lr, ny_lr);
            } else { // Fill non-masked outputs with zero
                outInitParams->fm_init[idx_hr] = 0.0;
                outInitParams->r2s_init[idx_hr] = 0.0;
            }
            // Clear amplitude outputs (will be filled by Stage 5)
            out_wr[idx_hr] = 0.0; out_wi[idx_hr] = 0.0;
            out_fr[idx_hr] = 0.0; out_fi[idx_hr] = 0.0;
        } // End r loop
    } // End c loop
    

    // --- NEW: Apply Gaussian Smoothing (as requested) ---
    std::cout << "  Applying Gaussian smoothing to upsampled maps..." << std::endl;
    
    // Map the 1D output arrays as 2D Eigen matrices
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> FM(outInitParams->fm_init, nx, ny);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> R2S(outInitParams->r2s_init, nx, ny);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> M(mask_vec_hr.data(), nx, ny);
    
    // Choose parameters
    // --- MODIFICATION: Increase passes for smoothness ---
    int radius = std::max(1, algoParams->SUBSAMPLE / 2); // e.g., factor=4 -> radius=2
    int passes = 3; // 3 passes = ~Gaussian (was 2)
    
    // Smooth fm (mask-aware, in-place)
    smooth_with_mask_inplace(FM, M, radius, passes);
    smooth_with_mask_inplace(R2S, M, radius, passes);
    
    std::cout << "  ...Smoothing complete." << std::endl;

    // --- NEW: Call Stage 5 at HIGH RESOLUTION (using 2D Maps) ---
    std::cout << "  Starting Stage 5: Final Amplitudes (High-Res, 2D)..." << std::endl;

    // Create maps for output amplitudes
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> WR(out_wr, nx, ny);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> WI(out_wi, nx, ny);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> FR(out_fr, nx, ny);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> FI(out_fi, nx, ny);
    
    stage5_final_amplitudes_cpp(
        WR, WI, FR, FI, // Output maps
        FM, R2S, M,     // Input maps
        imDataParams,   // Pointer to raw data
        t_n, 
        b_fat);
        
    std::cout << "  ...Stage 5 complete." << std::endl;

    std::cout << "--- Reconstruction Complete ---" << std::endl;
}