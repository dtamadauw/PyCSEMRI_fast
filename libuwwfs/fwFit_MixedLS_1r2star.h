#ifndef FWFIT_MIXEDLS_1R2STAR_FIXED_H
#define FWFIT_MIXEDLS_1R2STAR_FIXED_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include "param_str.h"
#include <Eigen/Core> // Eigen base
#include <unsupported/Eigen/NonLinearOptimization>

#define PI 3.14159265358979323846 // Use standard M_PI if available
#define GYRO 42.58
#define MAX_ITER 100 // Might not be needed by Eigen's LM

class fwFit_MixedLS_1r2star {

private:
    int nf;
    double *cursr;
    double *cursi;
    double *te;
    double *swr;
    double *swi;
    double *sfr;
    double *sfi;
    double *fF;
    double fieldStrength;
    double clockwise;
    double *initWr;
    double *initFr;
    double *initWi;
    double *initFi;
    double *initR2;
    double *initFieldmap;
    double *masksignal;
    algoParams_str *algoParams;
    initParams_str *initParams;
    imDataParams_str *imDataParams;
    int NUM_MAGN; // Number of magnitude echoes

public:
    int nte;
    int nx;
    int ny;
    double *outR2;
    double *outFieldmap;
    double *outWr;
    double *outWi;
    double *outFr;
    double *outFi;
    double *fitSr; // Added for fitted signal
    double *fitSi; // Added for fitted signal

    fwFit_MixedLS_1r2star() {} // Constructor

    // Destructor to free allocated memory
    ~fwFit_MixedLS_1r2star();

    // Member functions
    void initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in);
    void fit_all();
    void get_fitted_line(const Eigen::VectorXd &xval, Eigen::VectorXd &fval_magn, Eigen::MatrixXcd &fval_cplx); // Helper for fitted signal
};

#endif // FWFIT_MIXEDLS_1R2STAR_FIXED_H
