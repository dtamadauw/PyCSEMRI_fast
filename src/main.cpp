
#include <stdlib.h>
#include <stdio.h>
#include "fwFit_ComplexLS_1r2star.h"


int main(){

    
    int nx = 1;
    int ny = 1;
    double TE[6] = {0.0011, 0.0020, 0.0028, 0.0036, 0.0044, 0.0053};
    double images_r[6] = {3.2587,3.8639,3.9619,2.7412,3.3269,3.6598};
    double images_i[6] = {5.9425,4.3659,6.7439,5.4044,4.7778,3.7997};
    algoParams_str algoParams;
    initParams_str initParams;
    imDataParams_str imDataParams;


    printf("algoParams\n");
    //For algoParams
    algoParams.NUM_WAT_PEAKS = 1;
    algoParams.NUM_FAT_PEAKS = 6;
    algoParams.NUM_FMS = 101;

    algoParams.species_fat_amp[0] = 0.087;
    algoParams.species_fat_amp[1] = 0.693;
    algoParams.species_fat_amp[2] = 0.128;
    algoParams.species_fat_amp[3] = 0.004;
    algoParams.species_fat_amp[4] = 0.039;
    algoParams.species_fat_amp[5] = 0.048;

    algoParams.species_fat_freq[0] = -3.80;
    algoParams.species_fat_freq[1] = -3.40;
    algoParams.species_fat_freq[2] = -2.60; 
    algoParams.species_fat_freq[3] = -1.94; 
    algoParams.species_fat_freq[4] = -0.39; 
    algoParams.species_fat_freq[5] = 0.60;

    algoParams.species_wat_amp[0] = 1.0;
    algoParams.species_wat_freq[0] = 0.0;

    printf("initParams_str\n");
    //initParams_str
    double water_r_init[nx*ny] = {0.0};
    double fat_r_init[nx*ny] = {0.0};
    double water_i_init[nx*ny] = {0.0};
    double fat_i_init[nx*ny] = {0.0};
    double r2s_init[nx*ny] = {0.0};
    double fm_init[nx*ny] = {0.0};
    initParams.water_r_init = water_r_init;
    initParams.fat_r_init = fat_r_init;
    initParams.water_i_init = water_i_init;
    initParams.fat_i_init = fat_i_init;
    initParams.r2s_init = r2s_init;
    initParams.fm_init = fm_init;

    printf("imDataParams\n");
    //imDataParams
    for(int i=0;i<6;i++) imDataParams.TE[i] = TE[i];
    imDataParams.nte = 6;
    imDataParams.FieldStrength = 3.0;
    imDataParams.PrecessionIsClockwise = -1.0;
    imDataParams.images_r = images_r;
    imDataParams.images_i = images_i;
    imDataParams.im_dim[0] = nx;
    imDataParams.im_dim[1] = ny;
    

    printf("initialize_te\n");
    fwFit_ComplexLS_1r2star complex_fit;
    complex_fit.initialize_te(&imDataParams, &algoParams, &initParams);
    complex_fit.fit_all();


    printf("Wr: %f, Fr: %f, Wi: %f, Fi: %f, R2: %f, Field: %f\n", 
    complex_fit.outWr[0], complex_fit.outFr[0], complex_fit.outWi[0], complex_fit.outFi[0],
    complex_fit.outR2[0], complex_fit.outFieldmap[0]);

    return 0;
}