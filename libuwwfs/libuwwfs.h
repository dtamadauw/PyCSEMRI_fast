
#include <stdlib.h>
#include <stdio.h>
#include "param_str.h"


extern "C" {
void fwFit_ComplexLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams);
void fwFit_MixedLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams);
void fwFit_MagnLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams);
void fwFit_VarproLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams);
void VARPRO_LUT_c(imDataParams_str *imDataParams, algoParams_str *algoParams, outInitParams_str* outInitParams);
}
