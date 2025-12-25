#include "fwFit_MagnLS_1r2star.h"
#include <cmath> // For std::sqrt, std::exp, std::cos, std::sin
#include <complex> // For std::complex
#include <fstream>   // <<<<<<<<<<<< ADDED for file output
#include <iomanip>   // <<<<<<<<<<<< ADDED for formatting output

// Helper function to calculate the fitted magnitude signal
// Note: Input xval still has 6 parameters for consistency with other fits
void fwFit_MagnLS_1r2star::get_fitted_line_magn(const Eigen::VectorXd &xval, Eigen::VectorXd &fval) {
    double shatr;
    double Wr = xval(0);
    double Wi = xval(1);
    double Fr = xval(2);
    double Fi = xval(3);
    double r2 = xval(4);
    // double fieldmap = xval(5); // Fieldmap not used in magnitude model

    fval.resize(nte);
    for (int kt = 0; kt < nte; ++kt) {
        double EXP = exp(-te[kt] * r2);
        // Magnitude of complex sum: |(Wr+iWi)*c_w + (Fr+iFi)*c_f|
        // c_w = swr[kt] (assuming swi=0)
        // c_f = sfr[kt] + i*sfi[kt]
        // Real part = Wr*swr[kt] + Fr*sfr[kt] - Fi*sfi[kt]
        // Imag part = Wi*swr[kt] + Fr*sfi[kt] + Fi*sfr[kt]
        double real_part = Wr * swr[kt] + Fr * sfr[kt] - Fi*sfi[kt]; // Correct real part
        double imag_part = Wi * swr[kt] + Fr * sfi[kt] + Fi*sfr[kt]; // Correct imag part

        shatr = EXP * std::sqrt(real_part * real_part + imag_part * imag_part);
        fval(kt) = shatr;
    }
}


/**
 * @brief Functor for Eigen's LM solver for MAGNITUDE-ONLY fitting.
 * Uses the Jacobian from the original lsqcpp implementation.
 * Solves for 6 parameters for consistency: [Wr, Wi, Fr, Fi, R2*, fieldmap]
 * However, the Jacobian ignores Wi, Fi, and fieldmap based on the original code.
 */
struct MagnFitFunctorEigen {
    // Required by Eigen::LevenbergMarquardt
    enum {
        InputsAtCompileTime = Eigen::Dynamic, // p = 6
        ValuesAtCompileTime = Eigen::Dynamic  // n = nte (magnitude residuals)
    };

    // Member Data
    data_str *data;
    int m_inputs;
    int m_values;

    // Constructor
    MagnFitFunctorEigen(data_str *d, int nte) : data(d) {
        m_inputs = 6; // Keep 6 inputs for consistency
        m_values = nte; // nte magnitude residuals
    }

    // Required: Get dimensions
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    /**
     * @brief Computes the magnitude residual vector f(x).
     */
    int operator()(const Eigen::VectorXd &xval, Eigen::VectorXd &fval) const {
        double shatr, EXP;
        int nte = data->nte;
        double *cursr = data->cursr; // Contains measured magnitudes
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi; // Assumed 0
        double *sfr = data->sfr;
        double *sfi = data->sfi;

        double Wr = xval(0);
        double Wi = xval(1);
        double Fr = xval(2);
        double Fi = xval(3);
        double r2 = xval(4);
        // double fieldmap = xval(5); // Not used in model

        // fval.resize(m_values); // Not needed

        for (int kt = 0; kt < nte; ++kt) {
            EXP = exp(-te[kt] * r2);
            // Magnitude of complex sum: |(Wr+iWi)*c_w + (Fr+iFi)*c_f|
            double real_part = Wr * swr[kt] + Fr * sfr[kt] - Fi*sfi[kt];
            double imag_part = Wi * swr[kt] + Fr * sfi[kt] + Fi*sfr[kt];
            shatr = EXP * std::sqrt(real_part * real_part + imag_part * imag_part);
            fval(kt) = shatr - cursr[kt]; // Difference from measured magnitude
        }
        return 0; // Success
    }

    /**
     * @brief Computes the Jacobian J(x), replicating the original logic.
     * Note: Only computes derivatives wrt Wr, Fr, R2*. Others are zero.
     */
    int df(const Eigen::VectorXd &xval, Eigen::MatrixXd &jacobian) const {
        double shatr_inner, EXP;
        double curJ1, curJ2, curJ3; // Only these were calculated originally
        int nte = data->nte;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi; // Assumed 0
        double *sfr = data->sfr;
        double *sfi = data->sfi;

        double Wr = xval(0);
        double Wi = xval(1);
        double Fr = xval(2);
        double Fi = xval(3);
        double r2 = xval(4);
        // double fieldmap = xval(5); // Not used

        // jacobian.setZero(m_values, m_inputs); // Not needed

        for (int kt = 0; kt < nte; ++kt) {
            EXP = exp(-te[kt] * r2);
            double real_part = Wr * swr[kt] + Fr * sfr[kt] - Fi*sfi[kt];
            double imag_part = Wi * swr[kt] + Fr * sfi[kt] + Fi*sfr[kt];
            // Original shat calculation (inner magnitude + epsilon)
            shatr_inner = std::sqrt(real_part * real_part + imag_part * imag_part) + 1e-12;

            // d/dWr (Original implementation) - Simplified from complex version
            // Assumed d/dWr = EXP * real_part / shatr_inner approx?
            // Let's use the derivative copied from the original lsqcpp magnitude code
             curJ1 = EXP*(Wr*swr[kt] + Fr*sfr[kt])/shatr_inner; // From original ParabolicError_Magn
            jacobian(kt, 0) = curJ1;

            // d/dWi (Original was zero)
            jacobian(kt, 1) = 0.0;

            // d/dFr (Original implementation)
             curJ2 = EXP*(Fr*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) + Wr*sfr[kt])/shatr_inner; // From original ParabolicError_Magn
            jacobian(kt, 2) = curJ2;

            // d/dFi (Original was zero)
            jacobian(kt, 3) = 0.0;

            // d/dR2* (Original implementation)
            // Note: This term looks unusual, but we must match it.
            curJ3 = EXP*(-Wr*Wr*te[kt] - Wr*Fr*sfr[kt]*te[kt] -Fr*Fr*te[kt]*(sfr[kt]*sfr[kt] + sfi[kt]*sfi[kt]) - Wr*Fr*sfr[kt]*te[kt])/shatr_inner; // From original ParabolicError_Magn
            jacobian(kt, 4) = curJ3;

            // d/dfieldmap (Original was zero)
            jacobian(kt, 5) = 0.0;
        }
        return 0; // Success
    }
};

// Destructor
fwFit_MagnLS_1r2star::~fwFit_MagnLS_1r2star() {
    delete[] cursr;
    // delete[] cursi; // Removed cursi member
    delete[] sfr;
    delete[] sfi;
    delete[] swr;
    delete[] swi;
    delete[] te; // Added te
    delete[] fitSr;
    delete[] fitSi;
    delete[] fF;
    delete[] outR2;
    delete[] outFieldmap;
    delete[] outWr;
    delete[] outWi;
    delete[] outFr;
    delete[] outFi;
}

// initialize_te function - Corrected memory allocation
void fwFit_MagnLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in) {
    this->imDataParams = imDataParams_in;
    this->algoParams = algoParams_in;
    this->initParams = initParams_in;
    this->nte = imDataParams->nte;
    this->fieldStrength = imDataParams->FieldStrength;
    this->clockwise = imDataParams->PrecessionIsClockwise;
    this->nx = imDataParams->im_dim[0];
    this->ny = imDataParams->im_dim[1];
    // this->nf = std::max(nx, ny); // nf set correctly below

    cursr = new double[nte]; // For magnitude data
    // cursi = new double[nte]; // Don't need imaginary part for magnitude fit storage
    sfr = new double[nte];
    sfi = new double[nte];
    swr = new double[nte];
    swi = new double[nte];
    te = new double[nte]; // Need to allocate te

    double waterAmp = algoParams->species_wat_amp[0];
    double *relAmps = algoParams->species_fat_amp;
    double *fPPM = algoParams->species_fat_freq;
    nf = algoParams->NUM_FAT_PEAKS;
    fF = new double[nf]; // Use new[]
    for (int kf = 0; kf < nf; kf++) {
        fF[kf] = fPPM[kf] * GYRO * fieldStrength;
    }

    initWr = initParams->water_r_init;
    initFr = initParams->fat_r_init;
    initWi = initParams->water_i_init;
    initFi = initParams->fat_i_init;
    initR2 = initParams->r2s_init;
    initFieldmap = initParams->fm_init; // Still needed for initial guess consistency
    masksignal = initParams->masksignal_init;

    for (int kt = 0; kt < nte; kt++) { // Use kt instead of kf
        te[kt] = imDataParams->TE[kt];
    }

    size_t num_pixels = (size_t)nx * ny;
    outR2 = new double[num_pixels];
    outFieldmap = new double[num_pixels];
    outWr = new double[num_pixels];
    outWi = new double[num_pixels];
    outFr = new double[num_pixels];
    outFi = new double[num_pixels];
    fitSr = new double[num_pixels * nte]; // Allocate fitSr
    fitSi = new double[num_pixels * nte]; // Allocate fitSi

    // Initialize water/fat signal models (no change needed here)
    for (int kt = 0; kt < nte; kt++) {
        swr[kt] = waterAmp;
        swi[kt] = 0.0; // Assume water is real
        sfr[kt] = 0.0;
        sfi[kt] = 0.0;
        for (int kf = 0; kf < nf; kf++) {
            sfr[kt] = sfr[kt] + relAmps[kf] * cos(2 * PI * te[kt] * fF[kf]);
            sfi[kt] = sfi[kt] + relAmps[kf] * sin(2 * PI * te[kt] * fF[kf]);
        }
    }
}

// Rewritten fit_all using Eigen::LevenbergMarquardt
void fwFit_MagnLS_1r2star::fit_all() {

    printf("Starting fit_all() for Magnitude Fit with Eigen LM...\n");

    // --- SETUP ---
    data_str data;
    data.nte = nte;
    data.NUM_MAGN = nte; // All echoes are magnitude for this fit
    data.cursr = cursr;
    // data.cursi = nullptr; // Not used
    data.te = te;
    data.swr = swr;
    data.swi = swi;
    data.sfr = sfr;
    data.sfi = sfi;

    MagnFitFunctorEigen functor(&data, this->nte);
    Eigen::LevenbergMarquardt<MagnFitFunctorEigen> optimizer(functor);

    // Configure optimizer
    optimizer.parameters.maxfev = 50;
    optimizer.parameters.xtol = 1e-4;
    optimizer.parameters.ftol = 1e-4;

    Eigen::VectorXd initialGuess(6); // Still use 6 params for consistency
    Eigen::VectorXd fitted_vector(nte); // For final fitted signal

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;

    std::ofstream log_file;
    const char* log_filename = "magn_debug_log.txt";

    printf("Starting pixel loop...\n");
    size_t num_pixels = (size_t)nx * ny;

    // --- PIXEL LOOP ---
    for (int kx = 0; kx < nx; kx++) {
        for (int ky = 0; ky < ny; ky++) {
            size_t idx = (size_t)kx + (size_t)ky * nx;

            if (masksignal[idx] > 0.1) {
                // Get MAGNITUDE signal for the current voxel
                for (int kt = 0; kt < nte; kt++) {
                    size_t im_offset = (size_t)kt * num_pixels;
                    double re = imsr[idx + im_offset];
                    double im = imsi[idx + im_offset];
                    cursr[kt] = std::sqrt(re * re + im * im); // Store magnitude in cursr
                    // cursi is not needed
                }

                // Set initial guess (using all 6 params for consistency)
                initialGuess(0) = initWr[idx];
                initialGuess(1) = initWi[idx];
                initialGuess(2) = initFr[idx];
                initialGuess(3) = initFi[idx];
                initialGuess(4) = initR2[idx];
                initialGuess(5) = initFieldmap[idx]; // Keep fieldmap guess

                // Run the optimization
                Eigen::LevenbergMarquardtSpace::Status status = optimizer.minimize(initialGuess);
                // initialGuess is updated in-place

                // Store results
                outWr[idx] = initialGuess(0);
                outWi[idx] = initialGuess(1);
                outFr[idx] = initialGuess(2);
                outFi[idx] = initialGuess(3);
                outR2[idx] = initialGuess(4);
                outFieldmap[idx] = initialGuess(5); // Store the final fieldmap too


                if((kx==0) && (ky==0)){
                    printf("outWr, outWi, R2s, fm = %f, %f, %f, %f\n",initialGuess(0), initialGuess(1), initialGuess(4), initialGuess(5));
                
                // --- Log Final State for Target Pixel ---
                    log_file.open(log_filename, std::ios::trunc); // Overwrite
                    log_file << "Input Data (cursr):\n[ ";
                    for(int kt=0; kt<nte; ++kt) log_file << cursr[kt] << (kt == nte-1 ? " ]\n" : " ");
                    log_file << "Initial Guess:\n[ ";
                    for(int i=0; i<6; ++i) log_file << initialGuess(i) << (i == 5 ? " ]\n" : " ");
                    log_file << "Status: " << (int)status << " ";
                    switch(status) { // CORRECTED ENUM CASES for logging
                        case Eigen::Success: log_file << "(Success)\n"; break;
                        case Eigen::NumericalIssue: log_file << "(NumericalIssue)\n"; break;
                        case Eigen::NoConvergence: log_file << "(NoConvergence)\n"; break;
                        case Eigen::InvalidInput: log_file << "(InvalidInput)\n"; break;
                        case 4: log_file << "(Converged - RelativeErrorTooSmall? Status=4)\n"; break; // Explicitly catch Status 4
                        default: log_file << "(Unknown Status: " << (int)status << ")\n"; break;
                    }
                    log_file << "Iterations: " << optimizer.iter << "\n";       // Correct access
                    log_file << "Func Evals: " << optimizer.nfev << "\n";        // Correct access
                    log_file << "Jac Evals: " << optimizer.njev << "\n";        // Correct access
                    double final_fnorm = optimizer.fnorm;              // Correct access
                    double final_param_norm = initialGuess.norm();
                    log_file << "Final Fnorm (Residual Norm): " << final_fnorm << (std::isfinite(final_fnorm) ? "" : " (Non-finite!)") << "\n";
                    log_file << "Final Param Norm:            " << final_param_norm << (std::isfinite(final_param_norm) ? "" : " (Non-finite!)") << "\n";
                    log_file << "Final Params:\n[ ";
                    for(int i=0; i<6; ++i) log_file << initialGuess(i) << (i == 5 ? " ]\n" : " ");
                    log_file.close(); // Close the log file for this pixel
                    
                }



                // --- Calculate and store final fitted signal ---
                get_fitted_line_magn(initialGuess, fitted_vector);
                for (int kt = 0; kt < nte; ++kt) {
                    size_t fit_idx = idx + (size_t)kt * num_pixels;
                    fitSr[fit_idx] = fitted_vector(kt); // Store magnitude
                    fitSi[fit_idx] = 0.0;               // Magnitude has no phase
                }

            } else {
                // Masked out voxel
                outWr[idx] = initWr[idx];
                outWi[idx] = initWi[idx];
                outFr[idx] = initFr[idx];
                outFi[idx] = initFi[idx];
                outR2[idx] = initR2[idx];
                outFieldmap[idx] = 0.0;
                for (int kt = 0; kt < nte; ++kt) {
                    size_t fit_idx = idx + (size_t)kt * num_pixels;
                    fitSr[fit_idx] = 0.0;
                    fitSi[fit_idx] = 0.0;
                }
            }
        } // end ky loop
    } // end kx loop
    printf("...Pixel loop complete.\n");
}
