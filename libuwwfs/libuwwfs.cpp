
#include "libuwwfs.h"
#include <string.h>
#include "fwFit_ComplexLS_1r2star.h"
#include "fwFit_MagnLS_1r2star.h"
#include "fwFit_MixedLS_1r2star.h"


void fwFit_ComplexLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams){
    
    fwFit_ComplexLS_1r2star complex_fit;
    complex_fit.initialize_te(imDataParams, algoParams, initParams);
    //complex_fit.fit_all();
    complex_fit.fit_all();

    
    memcpy(outParams->r2starmap, complex_fit.outR2, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));
    memcpy(outParams->fm, complex_fit.outFieldmap, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));
    memcpy(outParams->wat_r_amp, complex_fit.outWr, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));
    memcpy(outParams->fat_r_amp, complex_fit.outFr, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));
    memcpy(outParams->wat_i_amp, complex_fit.outWi, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));
    memcpy(outParams->fat_i_amp, complex_fit.outFi, sizeof(double)*(complex_fit.nx)*(complex_fit.ny));

    /*
    for(int i=0;i<6;i++){
        printf("Echo[%d] = %f,", i, imDataParams->images_r[(complex_fit.nx)*(complex_fit.ny)*i]);
    }
    printf("\n");

    printf("Wr: %f, Fr: %f, Wi: %f, Fi: %f, R2: %f, Field: %f\n", 
    complex_fit.outWr[0], complex_fit.outFr[0], complex_fit.outWi[0], complex_fit.outFi[0],
    complex_fit.outR2[0], complex_fit.outFieldmap[0]);
    */

}



void fwFit_MixedLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams){
    
    fwFit_MixedLS_1r2star mixed_fit;
    mixed_fit.initialize_te(imDataParams, algoParams, initParams);
    mixed_fit.fit_all();

    
    memcpy(outParams->r2starmap, mixed_fit.outR2, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fm, mixed_fit.outFieldmap, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->wat_r_amp, mixed_fit.outWr, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fat_r_amp, mixed_fit.outFr, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->wat_i_amp, mixed_fit.outWi, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fat_i_amp, mixed_fit.outFi, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));

    /*
    for(int i=0;i<6;i++){
        printf("Echo[%d] = %f,", i, imDataParams->images_r[(mixed_fit.nx)*(mixed_fit.ny)*i]);
    }
    printf("\n");
    */

}

void fwFit_MagnLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams){
    
    fwFit_MagnLS_1r2star magn_fit;
    magn_fit.initialize_te(imDataParams, algoParams, initParams);
    magn_fit.fit_all();

    
    memcpy(outParams->r2starmap, magn_fit.outR2, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fm, magn_fit.outFieldmap, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->wat_r_amp, magn_fit.outWr, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fat_r_amp, magn_fit.outFr, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->wat_i_amp, magn_fit.outWi, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fat_i_amp, magn_fit.outFi, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));

    /*
    for(int i=0;i<6;i++){
        printf("Echo[%d] = %f,", i, imDataParams->images_r[(magn_fit.nx)*(magn_fit.ny)*i]);
    }
    printf("\n");

    printf("Wr: %f, Fr: %f, Wi: %f, Fi: %f, R2: %f, Field: %f\n", 
    magn_fit.outWr[0], magn_fit.outFr[0], magn_fit.outWi[0], magn_fit.outFi[0],
    magn_fit.outR2[0], magn_fit.outFieldmap[0]);
    */
    

}