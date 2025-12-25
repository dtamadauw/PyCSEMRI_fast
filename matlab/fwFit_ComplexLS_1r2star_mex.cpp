/*=================================================================
 * fwFit_VarproLS_1r2star_mex.cpp
 *
 * This is the MEX gateway function to bridge MATLAB with the
 * libuwwfs.so shared library. It handles data marshalling between
 * MATLAB's mxArray format and the C-style structs required by the library.
 *
 * To compile in MATLAB (from the MATLAB command window):
 * >> mex -I/path/to/your/headers -L/path/to/your/library -luwwfs fwFit_VarproLS_1r2star_mex.cpp
 *
 * For example, if your headers are in '../libuwwfs' and your library is in '../build':
 * >> mex -I../libuwwfs -L../build -luwwfs fwFit_VarproLS_1r2star_mex.cpp
 *=================================================================*/

#include "mex.h"
#include "libuwwfs.h" // Your C-style header for the .so library
#include <vector>
#include <cstring>   // For memcpy

// Helper function to get a field from a MATLAB struct
const mxArray* get_field(const mxArray* st, const char* field_name) {
    const mxArray* field = mxGetField(st, 0, field_name);
    if (!field) {
        mexErrMsgIdAndTxt("MATLAB:struct:fieldNotFound", "Field '%s' not found in input struct.", field_name);
    }
    return field;
}

// The main MEX function - this is the entry point from MATLAB
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // --- Check for proper number of inputs and outputs ---
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("MATLAB:nrhs", "Three input arguments required: (imData, algoParams, initParams).");
    }
    if (nlhs != 6) {
        mexErrMsgIdAndTxt("MATLAB:nlhs", "Six output arguments required: [wat_r, wat_i, fat_r, fat_i, r2, fm].");
    }

    // --- Get pointers to input MATLAB structs ---
    const mxArray *imData_mx = prhs[0];
    const mxArray *algo_mx = prhs[1];
    const mxArray *init_mx = prhs[2];

    // --- 1. Unpack imDataParams ---
    imDataParams_str imData_c;
    const mxArray *images_r_mx = get_field(imData_mx, "images_r");
    const mxArray *images_i_mx = get_field(imData_mx, "images_i");
    const mwSize *dims = mxGetDimensions(images_r_mx);
    imData_c.im_dim[0] = dims[0];
    imData_c.im_dim[1] = dims[1];
    
    imData_c.images_r = (double*)mxGetData(images_r_mx);
    imData_c.images_i = (double*)mxGetData(images_i_mx);

    const mxArray *TE_mx = get_field(imData_mx, "TE");
    imData_c.nte = mxGetNumberOfElements(TE_mx);
    if (imData_c.nte > 32) mexErrMsgIdAndTxt("MATLAB:tooManyTEs", "Number of echoes cannot exceed 32.");
    memcpy(imData_c.TE, mxGetPr(TE_mx), imData_c.nte * sizeof(double));
    
    imData_c.FieldStrength = mxGetScalar(get_field(imData_mx, "FieldStrength"));
    imData_c.PrecessionIsClockwise = mxGetScalar(get_field(imData_mx, "PrecessionIsClockwise"));

    // --- 2. Unpack algoParams ---
    algoParams_str algo_c;
    const mxArray *fat_amp_mx = get_field(algo_mx, "species_fat_amp");
    const mxArray *fat_freq_mx = get_field(algo_mx, "species_fat_freq");
    algo_c.NUM_FAT_PEAKS = mxGetNumberOfElements(fat_amp_mx);
    if (algo_c.NUM_FAT_PEAKS > 32) mexErrMsgIdAndTxt("MATLAB:tooManyFatPeaks", "Number of fat peaks cannot exceed 32.");
    memcpy(algo_c.species_fat_amp, mxGetPr(fat_amp_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
    memcpy(algo_c.species_fat_freq, mxGetPr(fat_freq_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
    
    algo_c.NUM_WAT_PEAKS = 1; // Hardcoded based on your model
    algo_c.species_wat_amp[0] = 1.0;
    algo_c.species_wat_freq[0] = 0.0;

    // --- 3. Unpack initParams ---
    initParams_str init_c;
    size_t num_pixels = (size_t)imData_c.im_dim[0] * imData_c.im_dim[1];
    
    // The C++ library doesn't use these for VARPRO, but we create them to match the struct
    std::vector<double> water_r_init(num_pixels, 0.0);
    std::vector<double> water_i_init(num_pixels, 0.0);
    std::vector<double> fat_r_init(num_pixels, 0.0);
    std::vector<double> fat_i_init(num_pixels, 0.0);
    
    init_c.water_r_init = water_r_init.data();
    init_c.water_i_init = water_i_init.data();
    init_c.fat_r_init = fat_r_init.data();
    init_c.fat_i_init = fat_i_init.data();
    init_c.r2s_init = (double*)mxGetData(get_field(init_mx, "r2s_init"));
    init_c.fm_init = (double*)mxGetData(get_field(init_mx, "fm_init"));
    init_c.masksignal_init = (double*)mxGetData(get_field(init_mx, "masksignal_init"));
    
    // --- 4. Create MATLAB outputs and populate outParams ---
    outParams_str out_c;
    plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // wat_r
    plhs[1] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // wat_i
    plhs[2] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // fat_r
    plhs[3] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // fat_i
    plhs[4] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // r2starmap
    plhs[5] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL); // fm
    
    out_c.wat_r_amp = (double*)mxGetData(plhs[0]);
    out_c.wat_i_amp = (double*)mxGetData(plhs[1]);
    out_c.fat_r_amp = (double*)mxGetData(plhs[2]);
    out_c.fat_i_amp = (double*)mxGetData(plhs[3]);
    out_c.r2starmap = (double*)mxGetData(plhs[4]);
    out_c.fm        = (double*)mxGetData(plhs[5]);

    // The library expects these pointers to be valid, even if we don't use the output
    std::vector<double> fit_r_dummy(num_pixels * imData_c.nte);
    std::vector<double> fit_i_dummy(num_pixels * imData_c.nte);
    out_c.fit_r_amp = fit_r_dummy.data();
    out_c.fit_i_amp = fit_i_dummy.data();
    
    // --- Call the C++ Library Function ---
    mexPrintf("Calling libuwwfs: fwFit_ComplexLS_1r2star_c...\n");
    fwFit_ComplexLS_1r2star_c(&imData_c, &algo_c, &init_c, &out_c);
    mexPrintf("...Call complete.\n");
}

