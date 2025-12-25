#include "fwFit_VarproLS_1r2star.h"
#include <iostream> // For debugging print statements
#include <complex>

struct VarproFunctor
{
    // --- Required by Eigen::LevenbergMarquardt ---
    enum {
        InputsAtCompileTime = Eigen::Dynamic, // Non-linear params (p=2)
        ValuesAtCompileTime = Eigen::Dynamic  // Residuals (n=nte*2)
    };

    // --- Member Data ---
    data_str *data;
    double r2_scale = 100.0; // Scaling factor for R2*
    int m_inputs;            // Number of non-linear parameters (p)
    int m_values;            // Number of residuals (n)

    // --- Workspace & State ---
    // These are matrices we compute in operator() and re-use in df().
    // They must be 'mutable' because operator() is const.
    mutable Eigen::MatrixXd A;       // Design matrix [n x 4]
    mutable Eigen::Vector4d rho;     // Linear parameters [4 x 1]
    mutable Eigen::VectorXd s_vec;   // Signal vector [n x 1]
    mutable Eigen::Matrix4d AtA;     // A^T * A [4 x 4]
    
    // Temporary matrices for Jacobian calculation
    mutable Eigen::MatrixXd J_alpha;   // [n x p]
    mutable Eigen::MatrixXd AtJ_alpha; // [4 x p]
    mutable Eigen::MatrixXd term;      // [4 x p]


    // --- Constructor ---
    VarproFunctor(data_str *d, int nte) : data(d) 
    {
        m_inputs = 2;       // p = 2 (r2_scaled, fieldmap)
        m_values = nte * 2; // n = 2 * nte

        // Pre-allocate all workspace matrices
        A.resize(m_values, 4);
        rho.resize(4);
        s_vec.resize(m_values);
        AtA.resize(4, 4);
        
        J_alpha.resize(m_values, m_inputs);
        AtJ_alpha.resize(4, m_inputs);
        term.resize(4, m_inputs);
    }

    // --- Required: Get dimensions ---
    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    /**
     * @brief Solves for linear params and updates state.
     * This is the core of VARPRO, called by operator().
     * It computes and stores A, s_vec, AtA, and rho.
     */
    void solve_linear_params(const Eigen::VectorXd& x_nl) const {
        double r2 = x_nl(0) * r2_scale;
        double fieldmap = x_nl(1);
        int nte = data->nte;

        for (int i = 0; i < nte; ++i) {
            double EXP = exp(-data->te[i] * r2);
            double CS = cos(2 * M_PI * fieldmap * data->te[i]);
            double SN = sin(2 * M_PI * fieldmap * data->te[i]);

            // Fill A matrix (design matrix)
            A(i, 0)       = EXP * (CS * data->swr[i] - SN * data->swi[i]);
            A(i + nte, 0) = EXP * (SN * data->swr[i] + CS * data->swi[i]);
            A(i, 1)       = EXP * (-CS * data->swi[i] - SN * data->swr[i]);
            A(i + nte, 1) = EXP * (-SN * data->swi[i] + CS * data->swr[i]);
            A(i, 2)       = EXP * (CS * data->sfr[i] - SN * data->sfi[i]);
            A(i + nte, 2) = EXP * (SN * data->sfr[i] + CS * data->sfi[i]);
            A(i, 3)       = EXP * (-CS * data->sfi[i] - SN * data->sfr[i]);
            A(i + nte, 3) = EXP * (-SN * data->sfi[i] + CS * data->sfr[i]);
        }

        // Fill s_vec (signal vector)
        for(int i=0; i<nte; ++i) {
            s_vec(i) = data->cursr[i];
            s_vec(i + nte) = data->cursi[i];
        }
        
        // Store AtA for re-use in df()
        AtA = A.transpose() * A;
        
        // Solve for and store linear parameters rho
        rho = AtA.ldlt().solve(A.transpose() * s_vec);
    }


    /**
     * @brief Required Function 1: Computes the residual vector f(x).
     * f(x) = A(x_nl) * rho_opt - s_vec
     */
    int operator()(const Eigen::VectorXd &x_nl, Eigen::VectorXd &fval) const
    {
        // 1. Solve for linear params and update mutable members A, rho, s_vec, AtA
        solve_linear_params(x_nl);
        
        // 2. Compute residual vector
        fval = A * rho - s_vec;
        
        return 0; // Success
    }

    /**
     * @brief Required Function 2: Computes the Jacobian J(x).
     * J(x) is the derivative of the residual w.r.t. NON-LINEAR params.
     */
    int df(const Eigen::VectorXd &x_nl, Eigen::MatrixXd &jacobian) const
    {
        // This function assumes operator() was just called, so
        // A, rho, s_vec, and AtA are all up-to-date.

        double r2 = x_nl(0) * r2_scale;
        double fieldmap = x_nl(1);
        int nte = data->nte;
        
        // 1. Calculate J_alpha: Jacobian of the *model* (A*rho) w.r.t. non-linear params
        for(int kt = 0; kt < nte; ++kt) {
            double EXP = exp(-data->te[kt]*r2);
            double CS = cos(2*M_PI*fieldmap*data->te[kt]);
            double SN = sin(2*M_PI*fieldmap*data->te[kt]);

            // Re-compute model_r and model_i (or could store them...)
            double model_r = A(kt,0)*rho(0) + A(kt,1)*rho(1) + A(kt,2)*rho(2) + A(kt,3)*rho(3);
            double model_i = A(kt+nte,0)*rho(0) + A(kt+nte,1)*rho(1) + A(kt+nte,2)*rho(2) + A(kt+nte,3)*rho(3);

            // Column 0: Derivatives of model wrt r2_scaled
            // d(model)/d(r2_scaled) = d(model)/d(r2) * d(r2)/d(r2_scaled) = (-t*model) * r2_scale
            J_alpha(kt, 0) = -data->te[kt] * model_r * r2_scale;
            J_alpha(kt + nte, 0) = -data->te[kt] * model_i * r2_scale;
            
            // Column 1: Derivatives of model wrt Fieldmap
            double term1 = (rho(0)*data->swr[kt] + rho(2)*data->sfr[kt] - rho(1)*data->swi[kt] - rho(3)*data->sfi[kt]);
            double term2 = (rho(0)*data->swi[kt] + rho(2)*data->sfi[kt] + rho(1)*data->swr[kt] + rho(3)*data->sfr[kt]);
            J_alpha(kt, 1) = -2*M_PI*data->te[kt]*(SN*EXP*term1 + CS*EXP*term2);
            J_alpha(kt+nte, 1) =  2*M_PI*data->te[kt]*(CS*EXP*term1 - SN*EXP*term2);
        }
        
        // 2. Calculate the final VARPRO Jacobian using Golub-Pereyra formula
        AtJ_alpha = A.transpose() * J_alpha;
        term = AtA.ldlt().solve(AtJ_alpha); // Use stored AtA
        
        jacobian = J_alpha - A * term;
        
        return 0; // Success
    }
};




// --- Class Methods for fwFit_VarproLS_1r2star ---
// Constructor and Destructor are unchanged.
fwFit_VarproLS_1r2star::fwFit_VarproLS_1r2star() {}
fwFit_VarproLS_1r2star::~fwFit_VarproLS_1r2star() {
    delete[] cursr; delete[] cursi; delete[] sfr; delete[] sfi;
    delete[] swr; delete[] swi; delete[] fitSr; delete[] fitSi;
    delete[] fF; delete[] outR2; delete[] outFieldmap;
    delete[] outWr; delete[] outWi; delete[] outFr; delete[] outFi;
}

// initialize_te is unchanged.
void fwFit_VarproLS_1r2star::initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in) {
    this->imDataParams = imDataParams_in; this->algoParams = algoParams_in; this->initParams = initParams_in;
    this->nte = imDataParams->nte; this->fieldStrength = imDataParams->FieldStrength;
    this->clockwise = imDataParams->PrecessionIsClockwise;
    this->nx = imDataParams->im_dim[0]; this->ny = imDataParams->im_dim[1];
    cursr = new double[nte]; cursi = new double[nte]; sfr = new double[nte]; sfi = new double[nte];
    swr = new double[nte]; swi = new double[nte]; te = new double[nte];
    nf = algoParams->NUM_FAT_PEAKS; fF = new double[nf];
    for(int kf=0;kf<nf;kf++) fF[kf] = algoParams->species_fat_freq[kf]*GYRO*fieldStrength;
    initR2 = initParams->r2s_init; initFieldmap = initParams->fm_init;
    masksignal = initParams->masksignal_init;
    for(int kf=0;kf<nte;kf++) te[kf] = imDataParams->TE[kf];
    outR2 = new double[nx*ny]; outFieldmap = new double[nx*ny];
    fitSr = new double[nx*ny*nte]; fitSi = new double[nx*ny*nte];
    outWr = new double[nx*ny]; outWi = new double[nx*ny];
    outFr = new double[nx*ny]; outFi = new double[nx*ny];
    double waterAmp = algoParams->species_wat_amp[0];
    for(int kt=0;kt<nte;kt++) {
        swr[kt] = waterAmp; swi[kt] = 0.0; sfr[kt] = 0.0; sfi[kt] = 0.0;
        for(int kf=0;kf<nf;kf++) {
            sfr[kt] += algoParams->species_fat_amp[kf]*cos(2*M_PI*te[kt]*fF[kf]);
            sfi[kt] += algoParams->species_fat_amp[kf]*sin(2*M_PI*te[kt]*fF[kf]);
        }
    }
}

// The UPDATED fit_all method
void fwFit_VarproLS_1r2star::fit_all() {

    data_str data;
    data.nte = nte; data.cursr = cursr; data.cursi = cursi;
    data.te = te; data.swr = swr; data.swi = swi;
    data.sfr = sfr; data.sfi = sfi;

    VarproFunctor functor(&data, this->nte);
    Eigen::LevenbergMarquardt<VarproFunctor> optimizer(functor);
    optimizer.parameters.maxfev = 10; // Max function evaluations
    optimizer.parameters.gtol = 1e-5; // Gradient tolerance
    optimizer.parameters.xtol = 1e-5; // Step tolerance

    Eigen::VectorXd initialGuess(2);

    double *imsr = imDataParams->images_r;
    double *imsi = imDataParams->images_i;

    for (int kx = 0; kx < nx; kx++) {
        for (int ky = 0; ky < ny; ky++) {
            int idx = kx + ky * nx;
            if (masksignal[idx] > 0.1) {
                for (int kt = 0; kt < nte; kt++) {
                    cursr[kt] = imsr[idx + kt * nx * ny];
                    cursi[kt] = clockwise > 0 ? imsi[idx + kt * nx * ny] : -imsi[idx + kt * nx * ny];
                }

                Eigen::VectorXd initialGuess(2);
                // --- SCALING FIX: Scale down the initial guess for R2* ---
                initialGuess(0) = initR2[idx] / functor.r2_scale;
                initialGuess(1) = initFieldmap[idx];

                optimizer.minimize(initialGuess);
                
                functor.solve_linear_params(initialGuess);

                // Store results
                outR2[idx] = initialGuess(0) * functor.r2_scale; // Un-scale R2*
                outFieldmap[idx] = initialGuess(1);
                outWr[idx] = functor.rho(0);
                outWi[idx] = functor.rho(1);
                outFr[idx] = functor.rho(2);
                outFi[idx] = functor.rho(3);

                Eigen::VectorXd s_model = functor.A * functor.rho;
                for (int kt = 0; kt < nte; kt++) {
                    fitSr[idx + kt*nx*ny] = s_model(kt);
                    fitSi[idx + kt*nx*ny] = s_model(kt + nte);
                }


            } else {
                int idx = kx + ky * nx;
                outR2[idx] = initR2[idx]; outFieldmap[idx] = initFieldmap[idx];
                outWr[idx] = 0; outWi[idx] = 0;
                outFr[idx] = 0; outFi[idx] = 0;
            }
        }
    }
}

