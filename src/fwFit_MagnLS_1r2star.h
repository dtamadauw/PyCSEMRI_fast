/*#include "stdafx.h"*/
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <algorithm>
#include "param_str.h"
#include "lsqcpp.hpp"



#define PI 3.14159265
#define GYRO 42.58
#define MAX_ITER 100





class fwFit_MagnLS_1r2star{

    private:
    int nte;
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
    int nx;
    int ny;
    double *outR2;
    double *outFieldmap;
    double *outWr;
    double *outWi;
    double *outFr;
    double *outFi;
    fwFit_MagnLS_1r2star(){
        
    }

    ~fwFit_MagnLS_1r2star(){
        delete[] cursr;
        delete[] cursi;
        delete[] sfr;
        delete[] sfi;
        delete[] swr;
        delete[] swi;
        //delete[] fre;
        //delete[] fim;
        delete[] fF;
        delete[] outR2;
        delete[] outFieldmap;
        delete[] outWr;
        delete[] outWi;
        delete[] outFr;
        delete[] outFi;
    }


    void initialize_te(imDataParams_str *imDataParams_in, algoParams_str *algoParams_in, initParams_str *initParams_in);
    void fit_all();


};
