/*=================================================================
 * fwFit_MixedLS_1r2star_mex.cpp
 *
 * MEX gateway for Mixed Magnitude/Complex fitting using libuwwfs.
 * Handles marshalling between MATLAB and C++ data structures.
 * Added extensive debugging print statements and validation.
 *=================================================================*/

#include "mex.h"
#include "libuwwfs.h" // C-style header for the .so library
#include <vector>
#include <cstring>   // For memcpy
#include <string>    // For error messages
#include <stdexcept> // For error handling
#include <cmath>     // For isnan, isinf

// Helper function to get a field safely and check type/complexity
const mxArray* get_field_checked(const mxArray* st, const char* field_name, const char* struct_name, bool require_double = true, bool require_real = true) {
    if (!st || !mxIsStruct(st)) {
         mexErrMsgIdAndTxt("MATLAB:invalidInput:NotStruct", "Input '%s' must be a struct.", struct_name);
    }
    mxArray* field = mxGetField(st, 0, field_name); // Use non-const mxArray* for potential modification checks if needed
    if (!field) {
        // Check if the field exists but is empty (which might be okay for optional inputs if C++ handles NULL)
        // However, based on previous errors, we require fields to exist and be non-empty for now.
        mexErrMsgIdAndTxt("MATLAB:invalidInput:fieldNotFound", "Required field '%s' not found or empty in struct '%s'.", field_name, struct_name);
    }
     if (mxIsEmpty(field)) {
         mexErrMsgIdAndTxt("MATLAB:invalidInput:fieldEmpty", "Required field '%s' in struct '%s' must not be empty.", field_name, struct_name);
     }
    if (require_double && !mxIsDouble(field)) {
         mexErrMsgIdAndTxt("MATLAB:invalidInput:wrongType", "Field '%s' in struct '%s' must be of type double (is %s).", field_name, struct_name, mxGetClassName(field));
    }
     if (require_real && mxIsComplex(field)) {
         mexErrMsgIdAndTxt("MATLAB:invalidInput:wrongComplexity", "Field '%s' in struct '%s' must be real (non-complex).", field_name, struct_name);
    }
    return field; // Return const mxArray*
}

// Helper function to get scalar double safely
double get_scalar_double_checked(const mxArray* st, const char* field_name, const char* struct_name) {
    const mxArray* field = get_field_checked(st, field_name, struct_name, true, true);
    if (mxGetNumberOfElements(field) != 1) {
         mexErrMsgIdAndTxt("MATLAB:invalidInput:notScalar", "Field '%s' in struct '%s' must be a scalar.", field_name, struct_name);
    }
    double value = mxGetScalar(field);
    // Check for NaN/Inf which might cause issues in C++ unexpectedly
    if (mxIsNaN(value) || mxIsInf(value)) {
         mexWarnMsgIdAndTxt("MATLAB:inputWarning:nonFiniteScalar", "Warning: Scalar field '%s' in struct '%s' is NaN or Inf.", field_name, struct_name);
    }
    return value;
}

// Helper function to get double array pointer safely
double* get_double_array_ptr(const mxArray* field, const char* field_name, const char* struct_name) {
     // Type checks done in get_field_checked
     double* ptr = mxGetPr(field);
     if (!ptr) {
         // This should ideally not happen if field is valid and non-empty
         mexErrMsgIdAndTxt("MATLAB:internalError:nullDataPtr", "Failed to get data pointer for field '%s' in struct '%s'.", field_name, struct_name);
     }
     // Optional: Add check for NaN/Inf in the first element as a sanity check
     if (mxGetNumberOfElements(field) > 0 && (mxIsNaN(ptr[0]) || mxIsInf(ptr[0]))) {
          mexWarnMsgIdAndTxt("MATLAB:inputWarning:nonFiniteData", "Warning: First element of field '%s' in struct '%s' is NaN or Inf.", field_name, struct_name);
     }
     return ptr;
}


// The main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexPrintf("\n--- Entering fwFit_MixedLS_1r2star_mex ---\n");

    // --- Check Inputs/Outputs ---
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("MATLAB:nrhs", "Three input arguments required: (imData, algoParams, initParams).");
    }
    if (nlhs != 8) {
        mexErrMsgIdAndTxt("MATLAB:nlhs", "Eight output arguments required: [wat_r, wat_i, fat_r, fat_i, r2, fm, fit_r, fit_i].");
    }

    const mxArray *imData_mx = prhs[0];
    const mxArray *algo_mx = prhs[1];
    const mxArray *init_mx = prhs[2];

    imDataParams_str imData_c;
    algoParams_str algo_c;
    initParams_str init_c;
    outParams_str out_c; // This struct will hold pointers to MATLAB output arrays

    size_t num_pixels = 0;
    int nTE_val = 0;
    mwSize nx_mw = 0, ny_mw = 0, nTE_mw = 0;

    try {
        // --- 1. Unpack imDataParams ---
        mexPrintf("Unpacking imData...\n");
        const mxArray *images_r_mx = get_field_checked(imData_mx, "images_r", "imData", true, true);
        const mxArray *images_i_mx = get_field_checked(imData_mx, "images_i", "imData", true, true);

        mwSize num_dims = mxGetNumberOfDimensions(images_r_mx);
        if (num_dims < 3) mexErrMsgIdAndTxt("MATLAB:invalidDimensions", "'imData.images_r' must have at least 3 dimensions (nx, ny, nTE).");
        const mwSize *dims = mxGetDimensions(images_r_mx);
        nx_mw = dims[0]; ny_mw = dims[1]; nTE_mw = dims[2];
        imData_c.im_dim[0] = static_cast<int>(nx_mw);
        imData_c.im_dim[1] = static_cast<int>(ny_mw);
        nTE_val = static_cast<int>(nTE_mw);
        num_pixels = nx_mw * ny_mw;
        if (num_pixels == 0 || nTE_val == 0) mexErrMsgIdAndTxt("MATLAB:invalidDimensions", "Image dimensions (nx, ny, nTE) must be positive.");

        if (mxGetNumberOfDimensions(images_i_mx)!=num_dims || mxGetDimensions(images_i_mx)[0]!=nx_mw || mxGetDimensions(images_i_mx)[1]!=ny_mw || mxGetDimensions(images_i_mx)[2]!=nTE_mw)
            mexErrMsgIdAndTxt("MATLAB:dimensionMismatch", "Dimensions of 'imData.images_i' do not match 'imData.images_r'.");

        imData_c.images_r = get_double_array_ptr(images_r_mx, "images_r", "imData");
        imData_c.images_i = get_double_array_ptr(images_i_mx, "images_i", "imData");
        mexPrintf("  images_r ptr: %p, images_i ptr: %p\n", (void*)imData_c.images_r, (void*)imData_c.images_i);
        mexPrintf("  images_r[0]=%g, images_i[0]=%g\n", imData_c.images_r[0], imData_c.images_i[0]); // Check first value

        const mxArray *TE_mx = get_field_checked(imData_mx, "TE", "imData", true, true);
        imData_c.nte = static_cast<int>(mxGetNumberOfElements(TE_mx));
        if (imData_c.nte != nTE_val) mexErrMsgIdAndTxt("MATLAB:dimensionMismatch", "Num echoes in 'imData.TE' (%d) != 3rd dim of images (%d).", imData_c.nte, nTE_val);
        if (imData_c.nte <= 0 || imData_c.nte > 32) mexErrMsgIdAndTxt("MATLAB:invalidNumTEs", "Number of echoes (%d) must be between 1 and 32.", imData_c.nte);
        memcpy(imData_c.TE, mxGetPr(TE_mx), imData_c.nte * sizeof(double));
        mexPrintf("  TE[0]=%g\n", imData_c.TE[0]);

        imData_c.FieldStrength = get_scalar_double_checked(imData_mx, "FieldStrength", "imData");
        imData_c.PrecessionIsClockwise = get_scalar_double_checked(imData_mx, "PrecessionIsClockwise", "imData");
        mexPrintf("...imData unpacked (nx=%d, ny=%d, nTE=%d).\n", imData_c.im_dim[0], imData_c.im_dim[1], imData_c.nte);

        // --- 2. Unpack algoParams ---
        mexPrintf("Unpacking algoParams...\n");
        const mxArray *fat_amp_mx = get_field_checked(algo_mx, "species_fat_amp", "algoParams", true, true);
        const mxArray *fat_freq_mx = get_field_checked(algo_mx, "species_fat_freq", "algoParams", true, true);
        algo_c.NUM_FAT_PEAKS = static_cast<int>(mxGetNumberOfElements(fat_amp_mx));
        if (algo_c.NUM_FAT_PEAKS <= 0 || algo_c.NUM_FAT_PEAKS > 32) mexErrMsgIdAndTxt("MATLAB:invalidNumFatPeaks", "Number of fat peaks (%d) must be between 1 and 32.", algo_c.NUM_FAT_PEAKS);
        if (mxGetNumberOfElements(fat_freq_mx) != algo_c.NUM_FAT_PEAKS) mexErrMsgIdAndTxt("MATLAB:dimensionMismatch", "Mismatch between fat amp (%d) and freq (%zu) counts.", algo_c.NUM_FAT_PEAKS, mxGetNumberOfElements(fat_freq_mx));
        memcpy(algo_c.species_fat_amp, mxGetPr(fat_amp_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
        memcpy(algo_c.species_fat_freq, mxGetPr(fat_freq_mx), algo_c.NUM_FAT_PEAKS * sizeof(double));
        mexPrintf("  fat_amp[0]=%g, fat_freq[0]=%g\n", algo_c.species_fat_amp[0], algo_c.species_fat_freq[0]);

        algo_c.NUM_WAT_PEAKS = 1; algo_c.species_wat_amp[0] = 1.0; algo_c.species_wat_freq[0] = 0.0;
        mexPrintf("...algoParams unpacked (%d fat peaks).\n", algo_c.NUM_FAT_PEAKS);

        // --- 3. Unpack initParams ---
        mexPrintf("Unpacking initParams...\n");
        const mxArray *wr_mx = get_field_checked(init_mx, "water_r_init", "initParams", true, true);
        const mxArray *wi_mx = get_field_checked(init_mx, "water_i_init", "initParams", true, true);
        const mxArray *fr_mx = get_field_checked(init_mx, "fat_r_init", "initParams", true, true);
        const mxArray *fi_mx = get_field_checked(init_mx, "fat_i_init", "initParams", true, true);
        const mxArray *r2_mx = get_field_checked(init_mx, "r2s_init", "initParams", true, true);
        const mxArray *fm_mx = get_field_checked(init_mx, "fm_init", "initParams", true, true);
        const mxArray *mask_mx = get_field_checked(init_mx, "masksignal_init", "initParams", true, true);

        if (mxGetNumberOfElements(wr_mx)!=num_pixels || mxGetNumberOfElements(wi_mx)!=num_pixels || mxGetNumberOfElements(fr_mx)!=num_pixels || mxGetNumberOfElements(fi_mx)!=num_pixels || mxGetNumberOfElements(r2_mx)!=num_pixels || mxGetNumberOfElements(fm_mx)!=num_pixels || mxGetNumberOfElements(mask_mx)!=num_pixels)
            mexErrMsgIdAndTxt("MATLAB:dimensionMismatch", "One or more initParams fields do not match image dimensions (nx*ny = %zu).", num_pixels);

        init_c.water_r_init = get_double_array_ptr(wr_mx, "water_r_init", "initParams");
        init_c.water_i_init = get_double_array_ptr(wi_mx, "water_i_init", "initParams");
        init_c.fat_r_init = get_double_array_ptr(fr_mx, "fat_r_init", "initParams");
        init_c.fat_i_init = get_double_array_ptr(fi_mx, "fat_i_init", "initParams");
        init_c.r2s_init = get_double_array_ptr(r2_mx, "r2s_init", "initParams");
        init_c.fm_init = get_double_array_ptr(fm_mx, "fm_init", "initParams");
        init_c.masksignal_init = get_double_array_ptr(mask_mx, "masksignal_init", "initParams");
        mexPrintf("  water_r_init ptr: %p, water_r_init[0]=%g\n", (void*)init_c.water_r_init, init_c.water_r_init[0]);
        mexPrintf("  r2s_init ptr: %p, r2s_init[0]=%g\n", (void*)init_c.r2s_init, init_c.r2s_init[0]);
        mexPrintf("...initParams unpacked.\n");

        // --- 4. Create MATLAB outputs and populate outParams ---
        mexPrintf("Creating output arrays (nx=%zu, ny=%zu, nTE=%d)...\n", nx_mw, ny_mw, imData_c.nte);
        mwSize dims2D[2] = {nx_mw, ny_mw};
        plhs[0] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL); plhs[1] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL);
        plhs[2] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL); plhs[3] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL);
        plhs[4] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL); plhs[5] = mxCreateNumericArray(2, dims2D, mxDOUBLE_CLASS, mxREAL);

        mwSize dims3D[3] = {nx_mw, ny_mw, nTE_mw};
        plhs[6] = mxCreateNumericArray(3, dims3D, mxDOUBLE_CLASS, mxREAL); plhs[7] = mxCreateNumericArray(3, dims3D, mxDOUBLE_CLASS, mxREAL);

        for (int i = 0; i < 8; ++i) if (!plhs[i]) mexErrMsgIdAndTxt("MATLAB:memoryError", "Failed to allocate memory for output array %d.", i+1);

        // Assign pointers AFTER creation using mxGetPr
        out_c.wat_r_amp = mxGetPr(plhs[0]); out_c.wat_i_amp = mxGetPr(plhs[1]);
        out_c.fat_r_amp = mxGetPr(plhs[2]); out_c.fat_i_amp = mxGetPr(plhs[3]);
        out_c.r2starmap = mxGetPr(plhs[4]); out_c.fm        = mxGetPr(plhs[5]);
        out_c.fit_r_amp = mxGetPr(plhs[6]); out_c.fit_i_amp = mxGetPr(plhs[7]);

        mexPrintf("  Output pointers assigned:\n");
        mexPrintf("    wat_r_amp: %p\n", (void*)out_c.wat_r_amp);
        mexPrintf("    fit_r_amp: %p\n", (void*)out_c.fit_r_amp);
        mexPrintf("...Output arrays created.\n");

        // --- Call the C++ Library Function ---
        mexPrintf(">>> Calling libuwwfs function: fwFit_MixedLS_1r2star_c...\n");
        // Flush MATLAB's IO buffer so we see messages if C++ crashes immediately
        mexEvalString("drawnow;");
        fwFit_MixedLS_1r2star_c(&imData_c, &algo_c, &init_c, &out_c);
        mexPrintf("<<< ...libuwwfs call complete.\n");

        // Check first element value AFTER call
        mexPrintf("  First pixel output wat_r: %g\n", out_c.wat_r_amp[0]);
        mexPrintf("  First pixel output r2: %g\n", out_c.r2starmap[0]);
        mexPrintf("  First pixel output fit_r[0]: %g\n", out_c.fit_r_amp[0]);
         // Check a later element of fit_r to ensure it's not all zeros
         if (num_pixels > 0 && imData_c.nte > 0) {
             mexPrintf("  First pixel output fit_r[last_echo]: %g\n", out_c.fit_r_amp[num_pixels * (imData_c.nte - 1)]);
         }


    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MATLAB:cppError:stdException", "Caught C++ std::exception: %s", e.what());
    } catch (...) {
        mexErrMsgIdAndTxt("MATLAB:cppError:unknown", "Caught unknown C++ exception during MEX execution.");
    }
    mexPrintf("--- Exiting fwFit_MixedLS_1r2star_mex normally ---\n");
}

