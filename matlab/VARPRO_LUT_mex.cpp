/*=================================================================
 * fast_ideal_mex.cpp
 *
 * This is the MEX gateway function to bridge MATLAB with the
 * libuwwfs.so shared library. It calls the C-style function
 * fast_ideal_initial_guess_c from the library.
 *=================================================================*/

#include "mex.h"
#include "libuwwfs.h" // Correct header for the shared library
#include <vector>
#include <cstring>   // For memcpy

// Helper function to get a field from a MATLAB struct
const mxArray* get_field(const mxArray* st, const char* field_name, int index = 0) {
    const mxArray* field = mxGetField(st, index, field_name);
    if (!field) {
        mexErrMsgIdAndTxt("MATLAB:struct:fieldNotFound", "Field '%s' not found in input struct.", field_name);
    }
    return field;
}

// The main MEX function - this is the entry point from MATLAB
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 2) mexErrMsgIdAndTxt("MATLAB:nrhs", "Two input arguments required: (imDataParams, algoParams).");
    if (nlhs != 7) mexErrMsgIdAndTxt("MATLAB:nlhs", "Seven output arguments required: [fm_init, r2s_init, mask, water_r, fat_r, water_i, fat_i].");

    const mxArray *imData_mx = prhs[0];
    const mxArray *algo_mx = prhs[1];

    // --- 1. Unpack imDataParams ---
    imDataParams_str imData_c;
    const mxArray *images_mx = get_field(imData_mx, "images");
    const mwSize *dims = mxGetDimensions(images_mx);
    imData_c.im_dim[0] = dims[0];
    imData_c.im_dim[1] = dims[1];
    
    size_t num_elements = mxGetNumberOfElements(images_mx);
    std::vector<double> images_r_vec(num_elements);
    std::vector<double> images_i_vec(num_elements);
    
    #if MX_HAS_INTERLEAVED_COMPLEX
        mxComplexDouble *complex_data = mxGetComplexDoubles(images_mx);
        for(size_t i=0; i < num_elements; ++i) {
            images_r_vec[i] = complex_data[i].real;
            images_i_vec[i] = complex_data[i].imag;
        }
    #else
        double *pr = mxGetPr(images_mx);
        double *pi = mxGetPi(images_mx);
        if(pi) { 
             memcpy(images_r_vec.data(), pr, num_elements * sizeof(double));
             memcpy(images_i_vec.data(), pi, num_elements * sizeof(double));
        } else {
             memcpy(images_r_vec.data(), pr, num_elements * sizeof(double));
        }
    #endif
    
    imData_c.images_r = images_r_vec.data();
    imData_c.images_i = images_i_vec.data();
    
    const mxArray *TE_mx = get_field(imData_mx, "TE");
    imData_c.nte = mxGetNumberOfElements(TE_mx);
    memcpy(imData_c.TE, mxGetPr(TE_mx), imData_c.nte * sizeof(double));
    imData_c.FieldStrength = mxGetScalar(get_field(imData_mx, "FieldStrength"));
    imData_c.PrecessionIsClockwise = mxGetScalar(get_field(imData_mx, "PrecessionIsClockwise"));

    // --- 2. Unpack algoParams ---
    algoParams_str algo_c;
    const mxArray* species_struct = get_field(algo_mx, "species");
    const mxArray *fat_amp_mx = get_field(species_struct, "relAmps", 1);
    const mxArray *fat_freq_mx = get_field(species_struct, "frequency", 1);
    algo_c.NUM_FAT_PEAKS = mxGetNumberOfElements(fat_amp_mx);
    memcpy(algo_c.species_fat_amp, mxGetPr(fat_amp_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
    memcpy(algo_c.species_fat_freq, mxGetPr(fat_freq_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
    
    algo_c.NUM_WAT_PEAKS = 1;
    algo_c.species_wat_amp[0] = 1.0;
    algo_c.species_wat_freq[0] = 0.0;
    
    memcpy(algo_c.range_fm, mxGetPr(get_field(algo_mx, "range_fm")), 2 * sizeof(double));
    algo_c.NUM_FMS = (int)mxGetScalar(get_field(algo_mx, "NUM_FMS"));
    memcpy(algo_c.range_r2star, mxGetPr(get_field(algo_mx, "range_r2star")), 2 * sizeof(double));
    algo_c.NUM_R2STARS = (int)mxGetScalar(get_field(algo_mx, "NUM_R2STARS"));
    algo_c.mu = mxGetScalar(get_field(algo_mx, "mu"));
    algo_c.mask_threshold = mxGetScalar(get_field(algo_mx, "mask_threshold"));
    // --- Read new SUBSAMPLE parameter ---
    algo_c.SUBSAMPLE = (int)mxGetScalar(get_field(algo_mx, "SUBSAMPLE"));

    // --- 3. Create MATLAB outputs ---
    outInitParams_str out_c;
    mwSize out_dims[2] = {(mwSize)dims[0], (mwSize)dims[1]};
    plhs[0] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[4] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[5] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);
    plhs[6] = mxCreateNumericArray(2, out_dims, mxDOUBLE_CLASS, mxREAL);

    out_c.fm_init = (double*)mxGetData(plhs[0]);
    out_c.r2s_init = (double*)mxGetData(plhs[1]);
    out_c.masksignal_init = (double*)mxGetData(plhs[2]);
    out_c.wat_r_amp = (double*)mxGetData(plhs[3]);
    out_c.fat_r_amp = (double*)mxGetData(plhs[4]);
    out_c.wat_i_amp = (double*)mxGetData(plhs[5]);
    out_c.fat_i_amp = (double*)mxGetData(plhs[6]);

    // --- Call the C++ Library Function ---
    mexPrintf("Calling libuwwfs: fast_ideal_initial_guess_c...\n");
    VARPRO_LUT_c(&imData_c, &algo_c, &out_c);
    mexPrintf("...Call complete.\n");
}

