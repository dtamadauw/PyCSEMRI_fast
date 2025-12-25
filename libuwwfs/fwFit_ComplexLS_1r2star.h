/*#include "stdafx.h"*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include "param_str.h"
#include <unsupported/Eigen/NonLinearOptimization>


#define PI 3.14159265
#define GYRO 42.58
#define MAX_ITER 100





class fwFit_ComplexLS_1r2star{

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


    public:
    int nte;
    int nx;
    int ny;
    double *fitSr;
    double *fitSi;    
    double *outR2;
    double *outFieldmap;
    double *outWr;
    double *outWi;
    double *outFr;
    double *outFi;
    fwFit_ComplexLS_1r2star(){
        
    }

    ~fwFit_ComplexLS_1r2star(){
        delete[] cursr;
        delete[] cursi;
        delete[] sfr;
        delete[] sfi;
        delete[] swr;
        delete[] swi;
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

    void fitted_line(const Eigen::VectorXd &xval, Eigen::VectorXd &fval);

    void initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in);
    void fit_all();


};
