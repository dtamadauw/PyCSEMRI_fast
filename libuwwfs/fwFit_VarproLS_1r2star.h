#ifndef FW_FIT_VARPRO_LS_1R2STAR_H
#define FW_FIT_VARPRO_LS_1R2STAR_H

#include "param_str.h"
#include "lsqcpp.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>


#define PI 3.14159265
#define GYRO 42.58

class fwFit_VarproLS_1r2star {
private:
    // Member variables to hold data and parameters, same as your other classes
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
    // Pointers to initial guess maps
    double *initR2;
    double *initFieldmap;
    double *masksignal;
    // Pointers to input structs
    algoParams_str *algoParams;
    initParams_str *initParams;
    imDataParams_str *imDataParams;

public:
    // Public member variables for dimensions and outputs
    int nte;
    int nx;
    int ny;
    // Output arrays
    double *fitSr;
    double *fitSi;
    double *outR2;
    double *outFieldmap;
    double *outWr;
    double *outWi;
    double *outFr;
    double *outFi;

    fwFit_VarproLS_1r2star();
    ~fwFit_VarproLS_1r2star();

    // Main public methods
    void initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in);
    void fit_all();
};

#endif // FW_FIT_VARPRO_LS_1R2STAR_H

