
#include "libuwwfs.h"
#include <string.h>
#include "fwFit_ComplexLS_1r2star.h"
#include "fwFit_MagnLS_1r2star.h"
#include "fwFit_MixedLS_1r2star.h"
#include "fwFit_VarproLS_1r2star.h"
#include "fast_wfs.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <complex>
#include <iostream>
#include <fstream> // For writing debug files
#include <chrono>  // For timing


// Define complex types for clarity
using Eigen::VectorXcd;
using Eigen::VectorXd;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;


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
    memcpy(outParams->fit_r_amp, complex_fit.fitSr, sizeof(double)*(complex_fit.nx)*(complex_fit.ny)*(complex_fit.nte));
    memcpy(outParams->fit_i_amp, complex_fit.fitSi, sizeof(double)*(complex_fit.nx)*(complex_fit.ny)*(complex_fit.nte));
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
    
    printf("\n--- Entered fwFit_MixedLS_1r2star_c ---\n");
    // Check received pointers from MEX
    printf("  Received input pointers:\n");
    printf("    imDataParams: %p (images_r: %p)\n", (void*)imDataParams, (void*)imDataParams->images_r);
    printf("    algoParams: %p (fat_amp: %p)\n", (void*)algoParams, (void*)algoParams->species_fat_amp);
    printf("    initParams: %p (water_r_init: %p, r2s_init: %p)\n", (void*)initParams, (void*)initParams->water_r_init, (void*)initParams->r2s_init);
    printf("  Received output pointers:\n");
    printf("    outParams: %p (r2starmap: %p, fm: %p, wat_r: %p, fit_r: %p)\n",
           (void*)outParams, (void*)outParams->r2starmap, (void*)outParams->fm, (void*)outParams->wat_r_amp, (void*)outParams->fit_r_amp);

    fwFit_MixedLS_1r2star mixed_fit; // Create C++ object instance
    printf("  Initializing MixedLS...\n");
    // Pass the received pointers directly
    mixed_fit.initialize_te(imDataParams, algoParams, initParams);
    printf("  ...MixedLS initialized (nx=%d, ny=%d, nTE=%d)\n", mixed_fit.nx, mixed_fit.ny, mixed_fit.nte);
    // Check if initialization seems correct
     if (mixed_fit.nx <= 0 || mixed_fit.ny <= 0 || mixed_fit.nte <= 0) {
        printf("ERROR: Invalid dimensions after initialize_te!\n");
        // Optional: fill outputs with NaN or identifiable error values
        return; // Exit early
     }

    printf("  Calling MixedLS fit_all...\n");
    mixed_fit.fit_all(); // Run the fitting algorithm
    printf("  ...MixedLS fit_all complete.\n");

    // Check results *before* memcpy
    printf("  Result check (first pixel):\n");
    if (mixed_fit.outR2) printf("    outR2[0] = %g\n", mixed_fit.outR2[0]); else printf("    outR2 is NULL!\n");
    if (mixed_fit.outFieldmap) printf("    outFieldmap[0] = %g\n", mixed_fit.outFieldmap[0]); else printf("    outFieldmap is NULL!\n");
    if (mixed_fit.outWr) printf("    outWr[0] = %g\n", mixed_fit.outWr[0]); else printf("    outWr is NULL!\n");
    if (mixed_fit.fitSr) printf("    fitSr[0] = %g\n", mixed_fit.fitSr[0]); else printf("    fitSr is NULL!\n");
    if (mixed_fit.fitSi) printf("    fitSi[0] = %g\n", mixed_fit.fitSi[0]); else printf("    fitSi is NULL!\n");
     // Check last element too
     size_t num_pixels = (size_t)mixed_fit.nx * mixed_fit.ny;
     if (num_pixels > 0 && mixed_fit.nte > 0 && mixed_fit.fitSr) {
         printf("    fitSr[last_element] = %g\n", mixed_fit.fitSr[num_pixels * mixed_fit.nte - 1]);
     }


    printf("  Performing memcpy to output pointers...\n");
    size_t fit_size = num_pixels * mixed_fit.nte;

    // Defensive memcpy: Check both source and destination pointers
    if(outParams->r2starmap && mixed_fit.outR2) memcpy(outParams->r2starmap, mixed_fit.outR2, sizeof(double)*num_pixels); else printf("WARN: Skipping r2starmap memcpy (dst:%p, src:%p)\n", (void*)outParams->r2starmap, (void*)mixed_fit.outR2);
    if(outParams->fm && mixed_fit.outFieldmap) memcpy(outParams->fm, mixed_fit.outFieldmap, sizeof(double)*num_pixels); else printf("WARN: Skipping fm memcpy (dst:%p, src:%p)\n", (void*)outParams->fm, (void*)mixed_fit.outFieldmap);
    if(outParams->wat_r_amp && mixed_fit.outWr) memcpy(outParams->wat_r_amp, mixed_fit.outWr, sizeof(double)*num_pixels); else printf("WARN: Skipping wat_r memcpy (dst:%p, src:%p)\n", (void*)outParams->wat_r_amp, (void*)mixed_fit.outWr);
    if(outParams->fat_r_amp && mixed_fit.outFr) memcpy(outParams->fat_r_amp, mixed_fit.outFr, sizeof(double)*num_pixels); else printf("WARN: Skipping fat_r memcpy (dst:%p, src:%p)\n", (void*)outParams->fat_r_amp, (void*)mixed_fit.outFr);
    if(outParams->wat_i_amp && mixed_fit.outWi) memcpy(outParams->wat_i_amp, mixed_fit.outWi, sizeof(double)*num_pixels); else printf("WARN: Skipping wat_i memcpy (dst:%p, src:%p)\n", (void*)outParams->wat_i_amp, (void*)mixed_fit.outWi);
    if(outParams->fat_i_amp && mixed_fit.outFi) memcpy(outParams->fat_i_amp, mixed_fit.outFi, sizeof(double)*num_pixels); else printf("WARN: Skipping fat_i memcpy (dst:%p, src:%p)\n", (void*)outParams->fat_i_amp, (void*)mixed_fit.outFi);
    if(outParams->fit_r_amp && mixed_fit.fitSr) memcpy(outParams->fit_r_amp, mixed_fit.fitSr, sizeof(double)*fit_size); else printf("WARN: Skipping fit_r memcpy (dst:%p, src:%p)\n", (void*)outParams->fit_r_amp, (void*)mixed_fit.fitSr);
    if(outParams->fit_i_amp && mixed_fit.fitSi) memcpy(outParams->fit_i_amp, mixed_fit.fitSi, sizeof(double)*fit_size); else printf("WARN: Skipping fit_i memcpy (dst:%p, src:%p)\n", (void*)outParams->fit_i_amp, (void*)mixed_fit.fitSi);

    printf("--- Exiting fwFit_MixedLS_1r2star_c ---\n");

    /*
    fwFit_MixedLS_1r2star mixed_fit;
    mixed_fit.initialize_te(imDataParams, algoParams, initParams);
    mixed_fit.fit_all();

    
    memcpy(outParams->r2starmap, mixed_fit.outR2, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fm, mixed_fit.outFieldmap, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->wat_r_amp, mixed_fit.outWr, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fat_r_amp, mixed_fit.outFr, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->wat_i_amp, mixed_fit.outWi, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fat_i_amp, mixed_fit.outFi, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny));
    memcpy(outParams->fit_r_amp, mixed_fit.fitSr, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny)*(mixed_fit.nte));
    memcpy(outParams->fit_i_amp, mixed_fit.fitSi, sizeof(double)*(mixed_fit.nx)*(mixed_fit.ny)*(mixed_fit.nte));

    for(int i=0;i<6;i++){
        printf("Echo[%d] = %f,", i, imDataParams->images_r[(mixed_fit.nx)*(mixed_fit.ny)*i]);
    }
    printf("\n");
    */

}

void fwFit_MagnLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams){
    
    printf("--- Entered fwFit_MagnLS_1r2star_c ---\n");
    fwFit_MagnLS_1r2star magn_fit;
    magn_fit.initialize_te(imDataParams, algoParams, initParams);
    printf("  Input check: images_r[0]=%f, images_i[0]=%g\n", imDataParams->images_r[100+256*100], imDataParams->images_i[100+256*100]);
    printf("  MagnLS initialized (nx=%d, ny=%d, nTE=%d)\n", magn_fit.nx, magn_fit.ny, magn_fit.nte);
    printf("  Calling MagnLS fit_all...\n");
    magn_fit.fit_all();
    printf("  ...MagnLS fit_all complete.\n");
    printf("  Result check: wat_r_amp[0]=%f, wat_i_amp[0]=%f\n", magn_fit.outWr[100+256*100], magn_fit.outWi[100+256*100]);
    printf("  Output pointers: r2starmap=%p, fit_r_amp=%p\n", (void*)outParams->r2starmap, (void*)outParams->fit_r_amp);


    //fwFit_MagnLS_1r2star magn_fit;
    //magn_fit.initialize_te(imDataParams, algoParams, initParams);
    //magn_fit.fit_all();

    
    memcpy(outParams->r2starmap, magn_fit.outR2, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fm, magn_fit.outFieldmap, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->wat_r_amp, magn_fit.outWr, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fat_r_amp, magn_fit.outFr, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->wat_i_amp, magn_fit.outWi, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fat_i_amp, magn_fit.outFi, sizeof(double)*(magn_fit.nx)*(magn_fit.ny));
    memcpy(outParams->fit_r_amp, magn_fit.fitSr, sizeof(double)*(magn_fit.nx)*(magn_fit.ny)*(magn_fit.nte));
    memcpy(outParams->fit_i_amp, magn_fit.fitSi, sizeof(double)*(magn_fit.nx)*(magn_fit.ny)*(magn_fit.nte));

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

void fwFit_VarproLS_1r2star_c(imDataParams_str *imDataParams, algoParams_str *algoParams, initParams_str *initParams, outParams_str* outParams){
    
    fwFit_VarproLS_1r2star varpro_fit;
    varpro_fit.initialize_te(imDataParams, algoParams, initParams);
    varpro_fit.fit_all();
    
    memcpy(outParams->r2starmap, varpro_fit.outR2, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->fm, varpro_fit.outFieldmap, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->wat_r_amp, varpro_fit.outWr, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->fat_r_amp, varpro_fit.outFr, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->wat_i_amp, varpro_fit.outWi, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->fat_i_amp, varpro_fit.outFi, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny));
    memcpy(outParams->fit_r_amp, varpro_fit.fitSr, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny)*(varpro_fit.nte));
    memcpy(outParams->fit_i_amp, varpro_fit.fitSi, sizeof(double)*(varpro_fit.nx)*(varpro_fit.ny)*(varpro_fit.nte));
}

void VARPRO_LUT_c(imDataParams_str *imDataParams, algoParams_str *algoParams, outInitParams_str* outInitParams){

    VARPRO_LUT(imDataParams, algoParams, outInitParams);

}