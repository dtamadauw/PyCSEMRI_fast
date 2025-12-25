#ifndef FWFIT_MAGNLS_1R2STAR_H
#define FWFIT_MAGNLS_1R2STAR_H

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include "param_str.h"
#include <Eigen/Core>
#include <unsupported/Eigen/NonLinearOptimization>

#define PI 3.14159265
#define GYRO 42.58
#define MAX_ITER 100

class fwFit_MagnLS_1r2star {

private:
    int nf;
    double *cursr; // Stores magnitude data
    double *cursi; // Unused for magnitude data, but keep allocation for consistency? Or remove? Removing for now.
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

public:
    int nte;
    int nx;
    int ny;
    double *fitSr; // For fitted magnitude signal
    double *fitSi; // Will be zeros for magnitude fit
    double *outR2;
    double *outFieldmap;
    double *outWr;
    double *outWi;
    double *outFr;
    double *outFi;

    fwFit_MagnLS_1r2star() {} // Constructor

    ~fwFit_MagnLS_1r2star(); // Destructor declaration

    // Helper to calculate final fitted magnitude signal
    void get_fitted_line_magn(const Eigen::VectorXd &xval, Eigen::VectorXd &fval);

    void initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in);
    void fit_all();
};

#endif // FWFIT_MAGNLS_1R2STAR_H
