#include "fwFit_MixedLS_1r2star.h" // Use the updated header
#include <cmath> // For std::atan, std::cos, std::sin, std::sqrt

// complexPhase function is unchanged
double complexPhase(double x, double y) {
    double curPhi = 0.0;
    if (x > 0) {
        curPhi = atan(y / x);
    } else if (x < 0) {
        if (y >= 0) {
            curPhi = atan(y / x) + PI;
        } else {
            curPhi = atan(y / x) - PI;
        }
    } else if (x == 0) {
        if (y >= 0) {
            curPhi = PI / 2;
        } else {
            curPhi = -PI / 2;
        }
    }
    return curPhi;
}


struct MixedFitFunctorEigen {
    // Required by Eigen::LevenbergMarquardt
    enum {
        InputsAtCompileTime = Eigen::Dynamic, // p=5
        ValuesAtCompileTime = Eigen::Dynamic  // n=2*nte-NUM_MAGN
    };

    // Member Data
    data_str *data;
    int m_inputs;
    int m_values;

    // Constructor
    MixedFitFunctorEigen(data_str *d, int nte, int num_magn) : data(d) {
        m_inputs = 5;                  // W, F, phi, r2, fieldmap
        m_values = 2 * nte - num_magn; // Residuals
    }

    // Required: Get dimensions
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    /**
     * @brief Computes the residual vector f(x).
     */
    int operator()(const Eigen::VectorXd &xval, Eigen::VectorXd &fval) const {
        double shat, shatr, shati, CS, SN, EXP;
        int nte = data->nte;
        double *cursr = data->cursr;
        double *cursi = data->cursi;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi; // Note: swi is assumed 0 for water
        double *sfr = data->sfr;
        double *sfi = data->sfi;
        int NUM_MAGN = data->NUM_MAGN;

        double W = xval(0);
        double F = xval(1);
        double phi = xval(2);
        double r2 = xval(3);
        double fieldmap = xval(4);

        // Part 1: Magnitude residual(s)
        for (int kt = 0; kt < NUM_MAGN; ++kt) {
            EXP = exp(-te[kt] * r2);
            // Calculate magnitude of complex sum: |W*c_w + F*c_f|
            // Assuming c_w = swr[kt] (since swi is 0)
            // c_f = sfr[kt] + i*sfi[kt]
            double real_part = W * swr[kt] + F * sfr[kt];
            double imag_part = F * sfi[kt]; // W*swi[kt] is 0
            shat = EXP * std::sqrt(real_part * real_part + imag_part * imag_part);
            fval(kt) = shat - std::sqrt(cursr[kt] * cursr[kt] + cursi[kt] * cursi[kt]);
        }

        // Part 2: Complex residuals
        for (int kt = NUM_MAGN; kt < nte; ++kt) {
            CS = std::cos(phi + 2 * PI * fieldmap * te[kt]);
            SN = std::sin(phi + 2 * PI * fieldmap * te[kt]);
            EXP = exp(-te[kt] * r2);
             // S_model = exp(-t*R2) * exp(i*(phi + 2*pi*fm*t)) * (W*c_w + F*c_f)
            // Let CommonPhase = exp(i*(phi + 2*pi*fm*t)) = CS + i*SN
            // Let FatWaterSum = (W*swr[kt] + F*sfr[kt]) + i*(W*swi[kt] + F*sfi[kt])
            // S_model_real = EXP * (CS * real(FatWaterSum) - SN * imag(FatWaterSum))
            // S_model_imag = EXP * (SN * real(FatWaterSum) + CS * imag(FatWaterSum))
            double fw_real = W * swr[kt] + F * sfr[kt];
            double fw_imag = W * swi[kt] + F * sfi[kt]; // swi is likely 0
            shatr = EXP * (CS * fw_real - SN * fw_imag);
            shati = EXP * (SN * fw_real + CS * fw_imag);
            fval(kt) = shatr - cursr[kt];
            fval(kt + nte - NUM_MAGN) = shati - cursi[kt];
        }
        return 0; // Success
    }

    /**
     * @brief Computes the Jacobian J(x) using correct derivatives.
     */
    int df(const Eigen::VectorXd &xval, Eigen::MatrixXd &jacobian) const {
        double shat, shatr, shati, CS, SN, EXP;
        double curJ1, curJ2, curJ3, curJ4, curJ5;
        double expr2, sinfm, cosfm;
        int nte = data->nte;
        double *te = data->te;
        double *swr = data->swr;
        double *swi = data->swi; // Assume 0
        double *sfr = data->sfr;
        double *sfi = data->sfi;
        int NUM_MAGN = data->NUM_MAGN;

        double W = xval(0);
        double F = xval(1);
        double phi = xval(2);
        double r2 = xval(3);
        double fieldmap = xval(4);

        // --- CORRECTED JACOBIAN ---
        // Part 1: Jacobian for Magnitude residual(s)
        for (int kt = 0; kt < NUM_MAGN; ++kt) {
            EXP = exp(-te[kt] * r2);

            // Calculate magnitude of complex sum: |W*c_w + F*c_f|
            double real_part = W * swr[kt] + F * sfr[kt];
            double imag_part = F * sfi[kt]; // W*swi[kt] is 0
            shat = std::sqrt(real_part * real_part + imag_part * imag_part);
            double shat_inv = 1.0 / (shat + 1e-12); // Add epsilon for stability

            // Model = EXP * shat
            // d(Model)/d(param) = (d(EXP)/d(param) * shat) + (EXP * d(shat)/d(param))

            // d(shat)/dW = (real_part*swr[kt] + imag_part*swi[kt/0]) * shat_inv
            // Since swi=0, d(shat)/dW = (real_part*swr[kt]) * shat_inv
            curJ1 = EXP * (real_part * swr[kt]) * shat_inv;
            jacobian(kt, 0) = curJ1;

            // d(shat)/dF = (real_part*sfr[kt] + imag_part*sfi[kt]) * shat_inv
            curJ2 = EXP * (real_part * sfr[kt] + imag_part * sfi[kt]) * shat_inv;
            jacobian(kt, 1) = curJ2;

            // d(shat)/dPhi = 0
            jacobian(kt, 2) = 0.0;

            // d(Model)/dr2 = d(EXP)/dr2 * shat = -te[kt]*EXP * shat
            curJ4 = -te[kt] * EXP * shat;
            jacobian(kt, 3) = curJ4;

            // d(shat)/dfieldmap = 0
            jacobian(kt, 4) = 0.0;
        }
        // --- END CORRECTED JACOBIAN PART 1 ---


        // Part 2: Jacobian for Complex residuals (Unchanged, was correct)
        for (int kt = NUM_MAGN; kt < nte; ++kt) {
            expr2 = exp(-te[kt] * r2);
            double phase_term = phi + 2 * PI * fieldmap * te[kt];
            sinfm = std::sin(phase_term);
            cosfm = std::cos(phase_term);
            double fw_real = W * swr[kt] + F * sfr[kt];
            double fw_imag = W * swi[kt] + F * sfi[kt]; // swi is likely 0
            shatr = expr2 * (cosfm * fw_real - sinfm * fw_imag);
            shati = expr2 * (sinfm * fw_real + cosfm * fw_imag);

            // d(Real, Imag) / dW
            // d(fw_real)/dW = swr[kt], d(fw_imag)/dW = swi[kt]
            curJ1 = expr2 * (cosfm * swr[kt] - sinfm * swi[kt]); // d(shatr)/dW
            jacobian(kt, 0) = curJ1;
            curJ1 = expr2 * (sinfm * swr[kt] + cosfm * swi[kt]); // d(shati)/dW
            jacobian(kt + nte - NUM_MAGN, 0) = curJ1;
            // d(Real, Imag) / dF
            // d(fw_real)/dF = sfr[kt], d(fw_imag)/dF = sfi[kt]
            curJ2 = expr2 * (cosfm * sfr[kt] - sinfm * sfi[kt]); // d(shatr)/dF
            jacobian(kt, 1) = curJ2;
            curJ2 = expr2 * (sinfm * sfr[kt] + cosfm * sfi[kt]); // d(shati)/dF
            jacobian(kt + nte - NUM_MAGN, 1) = curJ2;
            // d(Real, Imag) / dPhi
            // d(cosfm)/dPhi = -sinfm, d(sinfm)/dPhi = cosfm
            curJ3 = expr2 * (-sinfm * fw_real - cosfm * fw_imag); // d(shatr)/dPhi
            jacobian(kt, 2) = curJ3;
            curJ3 = expr2 * (cosfm * fw_real - sinfm * fw_imag); // d(shati)/dPhi
            jacobian(kt + nte - NUM_MAGN, 2) = curJ3;
            // d(Real, Imag) / dr2
            // d(expr2)/dr2 = -te[kt]*expr2
            curJ4 = -te[kt] * shatr; // d(shatr)/dr2
            jacobian(kt, 3) = curJ4;
            curJ4 = -te[kt] * shati; // d(shati)/dr2
            jacobian(kt + nte - NUM_MAGN, 3) = curJ4;
            // d(Real, Imag) / dfieldmap
            // d(cosfm)/dfm = -sinfm * 2*PI*te[kt], d(sinfm)/dfm = cosfm * 2*PI*te[kt]
            double factor = 2 * PI * te[kt];
            curJ5 = expr2 * (-sinfm * fw_real - cosfm * fw_imag) * factor; // d(shatr)/dfm
            jacobian(kt, 4) = curJ5;
            curJ5 = expr2 * (cosfm * fw_real - sinfm * fw_imag) * factor; // d(shati)/dfm
            jacobian(kt + nte - NUM_MAGN, 4) = curJ5;
        }
        return 0; // Success
    }
};


// Destructor - Ensure all allocated memory is deleted
fwFit_MixedLS_1r2star::~fwFit_MixedLS_1r2star() {
    delete[] cursr;
    delete[] cursi;
    delete[] sfr;
    delete[] sfi;
    delete[] swr;
    delete[] swi;
    delete[] te;
    delete[] fF; // Correctly matches new[]
    delete[] outR2;
    delete[] outFieldmap;
    delete[] outWr;
    delete[] outWi;
    delete[] outFr;
    delete[] outFi;
    delete[] fitSr; // Delete added members
    delete[] fitSi; // Delete added members
}

// Helper to calculate the final fitted signal (magnitude and complex parts)
void fwFit_MixedLS_1r2star::get_fitted_line(const Eigen::VectorXd &xval, Eigen::VectorXd &fval_magn, Eigen::MatrixXcd &fval_cplx)
{
    double shat, shatr, shati, CS, SN, EXP;
    double W = xval(0);
    double F = xval(1);
    double phi = xval(2);
    double r2 = xval(3);
    double fieldmap = xval(4);

    fval_magn.resize(NUM_MAGN);
    fval_cplx.resize(nte - NUM_MAGN, 1); // Store complex values directly

    // Part 1: Magnitude
    for (int kt = 0; kt < NUM_MAGN; ++kt) {
        EXP = exp(-te[kt] * r2);
        shat = EXP * std::sqrt((W * swr[kt] + F * sfr[kt]) * (W * swr[kt] + F * sfr[kt]) + (W * swi[kt] + F * sfi[kt]) * (W * swi[kt] + F * sfi[kt]));
        fval_magn(kt) = shat;
    }
    // Part 2: Complex
    for (int kt = NUM_MAGN; kt < nte; ++kt) {
        CS = std::cos(phi + 2 * PI * fieldmap * te[kt]);
        SN = std::sin(phi + 2 * PI * fieldmap * te[kt]);
        EXP = exp(-te[kt] * r2);
        shatr = CS * EXP * (W * swr[kt] + F * sfr[kt]) - SN * EXP * (W * swi[kt] + F * sfi[kt]);
        shati = SN * EXP * (W * swr[kt] + F * sfr[kt]) + CS * EXP * (W * swi[kt] + F * sfi[kt]);
        fval_cplx(kt - NUM_MAGN) = std::complex<double>(shatr, shati);
    }
}


// initialize_te function - Corrected memory allocation and adds fitSr/fitSi
void fwFit_MixedLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in) {
    this->imDataParams = imDataParams_in;
    this->algoParams = algoParams_in;
    this->initParams = initParams_in;
    this->nte = imDataParams->nte;
    this->fieldStrength = imDataParams->FieldStrength;
    this->clockwise = imDataParams->PrecessionIsClockwise;
    this->nx = imDataParams->im_dim[0];
    this->ny = imDataParams->im_dim[1];
    this->nf = std::max(nx, ny); // nf seems unused later? Retained for consistency
    this->NUM_MAGN = 1;         // Hardcoded as per original

    // Allocate memory using new[] consistently
    cursr = new double[nte];
    cursi = new double[nte];
    sfr = new double[nte];
    sfi = new double[nte];
    swr = new double[nte];
    swi = new double[nte];
    te = new double[nte];

    double waterAmp = algoParams->species_wat_amp[0];
    double *relAmps = algoParams->species_fat_amp;
    double *fPPM = algoParams->species_fat_freq;
    nf = algoParams->NUM_FAT_PEAKS; // Use the correct number of peaks
    fF = new double[nf];            // Allocate with new[]
    for (int kf = 0; kf < nf; kf++) {
        fF[kf] = fPPM[kf] * GYRO * fieldStrength;
    }

    initWr = initParams->water_r_init;
    initFr = initParams->fat_r_init;
    initWi = initParams->water_i_init;
    initFi = initParams->fat_i_init;
    initR2 = initParams->r2s_init;
    initFieldmap = initParams->fm_init;
    masksignal = initParams->masksignal_init;

    for (int kt = 0; kt < nte; kt++) {
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

    // Initialize water/fat signal models
    // No need to reallocate or recalculate fF here
    for (int kt = 0; kt < nte; kt++) {
        swr[kt] = waterAmp;
        swi[kt] = 0.0;
        sfr[kt] = 0.0;
        sfi[kt] = 0.0;
        for (int kf = 0; kf < nf; kf++) {
            sfr[kt] = sfr[kt] + relAmps[kf] * cos(2 * PI * te[kt] * fF[kf]);
            sfi[kt] = sfi[kt] + relAmps[kf] * sin(2 * PI * te[kt] * fF[kf]);
        }
    }
}


// Rewritten fit_all using Eigen::LevenbergMarquardt
void fwFit_MixedLS_1r2star::fit_all() {

    printf("Starting fit_all() with Eigen LM...\n");

    // --- SETUP ---
    data_str data; // Struct to pass persistent data to functor
    data.nte = nte;
    data.NUM_MAGN = this->NUM_MAGN;
    data.cursr = cursr; // Pointer to pixel's data (filled in loop)
    data.cursi = cursi; // Pointer to pixel's data (filled in loop)
    data.te = te;       // Pointer to echo times (constant)
    data.swr = swr;     // Pointer to water signal model (constant)
    data.swi = swi;     // Pointer to water signal model (constant)
    data.sfr = sfr;     // Pointer to fat signal model (constant)
    data.sfi = sfi;     // Pointer to fat signal model (constant)

    MixedFitFunctorEigen functor(&data, this->nte, this->NUM_MAGN);
    Eigen::LevenbergMarquardt<MixedFitFunctorEigen> optimizer(functor);

    // Configure optimizer (adjust tolerances as needed)
    optimizer.parameters.maxfev = 25; // Max function evaluations
    optimizer.parameters.xtol = 1e-3; // Step tolerance (change in parameters)
    optimizer.parameters.ftol = 1e-3; // Residual tolerance (change in error)
    // optimizer.parameters.gtol = 1e-4; // Gradient tolerance (not used by default LM)

    Eigen::VectorXd initialGuess(5);          // Pre-allocate guess vector [W, F, phi, r2, fieldmap]
    Eigen::VectorXd fitted_magn_vals(NUM_MAGN);      // For final fitted signal
    Eigen::MatrixXcd fitted_cplx_vals(nte - NUM_MAGN, 1); // For final fitted signal

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;
    double curPhi, curAmpW, curAmpF;

    printf("Starting pixel loop...\n");
    size_t num_pixels = (size_t)nx * ny;

    // --- PIXEL LOOP ---
    for (int kx = 0; kx < nx; kx++) {
        for (int ky = 0; ky < ny; ky++) {
            size_t idx = (size_t)kx + (size_t)ky * nx; // Use size_t for index
            size_t base_idx_im = idx;

            if (masksignal[idx] > 0.1) {
                // Get signal for the current voxel
                for (int kt = 0; kt < nte; kt++) {
                    size_t im_offset = (size_t)kt * num_pixels;
                    cursr[kt] = imsr[base_idx_im + im_offset];
                    cursi[kt] = clockwise > 0 ? imsi[base_idx_im + im_offset] : -imsi[base_idx_im + im_offset];
                }

                // Set initial guess (using the validated logic)
                curAmpW = std::sqrt(initWr[idx] * initWr[idx] + initWi[idx] * initWi[idx]);
                curAmpF = std::sqrt(initFr[idx] * initFr[idx] + initFi[idx] * initFi[idx]);
                // Original logic for initial phase:
                // if(curAmpW>curAmpF) {
                //     curPhi = complexPhase(initWr[idx],initWi[idx]);
                // } else {
                //     curPhi = complexPhase(initFr[idx],initFi[idx]);
                // }
                // Simplified/validated logic:
                curPhi = complexPhase(initWr[idx] + initFr[idx], initWi[idx] + initFi[idx]);

                initialGuess(0) = curAmpW;
                initialGuess(1) = curAmpF;
                initialGuess(2) = curPhi;
                initialGuess(3) = initR2[idx];
                initialGuess(4) = initFieldmap[idx];

                // Run the optimization
                Eigen::LevenbergMarquardtSpace::Status status = optimizer.minimize(initialGuess);
                // initialGuess is updated in-place

                // Store results
                curAmpW = initialGuess(0);
                curAmpF = initialGuess(1);
                curPhi = initialGuess(2);
                outR2[idx] = initialGuess(3);
                outFieldmap[idx] = initialGuess(4);
                outWr[idx] = curAmpW * std::cos(curPhi);
                outWi[idx] = curAmpW * std::sin(curPhi);
                outFr[idx] = curAmpF * std::cos(curPhi);
                outFi[idx] = curAmpF * std::sin(curPhi);

                // --- Calculate and store final fitted signal ---
                get_fitted_line(initialGuess, fitted_magn_vals, fitted_cplx_vals);
                for (int kt = 0; kt < NUM_MAGN; ++kt) {
                    size_t fit_idx = idx + (size_t)kt * num_pixels;
                    fitSr[fit_idx] = fitted_magn_vals(kt); // Store magnitude
                    fitSi[fit_idx] = 0.0;                  // Magnitude has no phase
                }
                for (int kt = NUM_MAGN; kt < nte; ++kt) {
                    size_t fit_idx = idx + (size_t)kt * num_pixels;
                    fitSr[fit_idx] = fitted_cplx_vals(kt - NUM_MAGN).real();
                    fitSi[fit_idx] = fitted_cplx_vals(kt - NUM_MAGN).imag();
                }

            } else {
                // Masked out voxel: Store initial values or zeros
                outWr[idx] = initWr[idx]; 
                outWi[idx] = initWi[idx]; 
                outFr[idx] = initFr[idx]; 
                outFi[idx] = initFi[idx]; 
                outR2[idx] = initR2[idx]; 
                outFieldmap[idx] = 0.0;   
                // Set fitted signal to zero for masked voxels
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
