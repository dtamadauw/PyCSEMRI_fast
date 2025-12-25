#ifndef _H_PARAM_STR_
#define _H_PARAM_STR_

#include <cmath> 


struct data_str{

    int nte;
    int NUM_MAGN;
    double *cursr;
    double *cursi;
    double *te;
    double *swr;
    double *swi;
    double *sfr;
    double *sfi;

};

struct algoParams_str{
    double species_wat_amp[32];
    double species_wat_freq[32];
    double species_fat_amp[32];
    double species_fat_freq[32];
    int NUM_WAT_PEAKS;
    int NUM_FAT_PEAKS;
    double range_fm[2];
    int NUM_FMS;
    double range_r2star[2];
    int NUM_R2STARS;
    double mu; // Spatial regularization weight
    double mask_threshold;
    int SUBSAMPLE; // subsampling
};

struct initParams_str{
    double *water_r_init;
    double *fat_r_init;
    double *water_i_init;
    double *fat_i_init;
    double *r2s_init;
    double *fm_init;
    double *masksignal_init;
};

struct imDataParams_str{
    double TE[32];
    int nte;
    double FieldStrength;
    double PrecessionIsClockwise;
    double *images_r;
    double *images_i;
    int im_dim[2];

};


struct outParams_str{
    double *r2starmap;
    double *fm;
    double *wat_r_amp;
    double *fat_r_amp;
    double *wat_i_amp;
    double *fat_i_amp;
    double *fit_r_amp;
    double *fit_i_amp;
};

struct outInitParams_str{
    double *r2s_init;
    double *fm_init;
    double *masksignal_init;
    double *wat_r_amp;
    double *fat_r_amp;
    double *wat_i_amp;
    double *fat_i_amp;
};

#endif	// _H_PARAM_STR_
