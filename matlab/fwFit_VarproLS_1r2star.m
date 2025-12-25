function outParams = fwFit_VarproLS_1r2star(imDataParams, algoParams, initParams)
%FWFIT_VARPROLS_1R2STAR MATLAB wrapper for the C++ VARPRO fitting library.
%
%   outParams = fwFit_VarproLS_1r2star(imDataParams, algoParams, initParams)
%
%   This function takes MATLAB structs as input, validates them, and calls
%   the compiled MEX gateway function to execute the fast C++ fitting.
%
%   INPUTS:
%       imDataParams - Struct with image data and acquisition parameters.
%           .images: [nx, ny, nTE] complex double array.
%           .TE: [1, nTE] double array of echo times in seconds.
%           .FieldStrength: Scalar, in Tesla.
%           .PrecessionIsClockwise: Scalar, typically -1 or 1.
%
%       algoParams - Struct with algorithm settings.
%           .species(1).name = 'water', .frequency, .relAmps
%           .species(2).name = 'fat', .frequency, .relAmps
%
%       initParams - Struct with initial guesses for the fit.
%           .r2s_init: [nx, ny] double array.
%           .fm_init: [nx, ny] double array (field map in Hz).
%           .masksignal_init: [nx, ny] double array (binary mask).
%
%   OUTPUTS:
%       outParams - Struct containing the fitted parameter maps.
%           .r2starmap, .fm, .water_amp (complex), .fat_amp (complex)
%

% --- Input Validation ---
if ~isfield(imDataParams, 'images') || ~isfield(imDataParams, 'TE')
    error('imDataParams must contain .images and .TE fields.');
end
if ~isfield(initParams, 'fieldmap') || ~isfield(initParams, 'r2starmap')
    error('initParams must contain .fieldmap and .r2starmap fields.');
end

% --- Prepare structs for MEX function ---
% The MEX function expects separate real/imaginary parts and specific field names.
imData.images_r = real(imDataParams.images);
imData.images_i = imag(imDataParams.images);
imData.TE = imDataParams.TE;
imData.FieldStrength = imDataParams.FieldStrength;
imData.PrecessionIsClockwise = imDataParams.PrecessionIsClockwise;

algo.species_fat_amp = algoParams.species(2).relAmps;
algo.species_fat_freq = algoParams.species(2).frequency;

init.r2s_init = initParams.r2starmap;
init.fm_init = initParams.fieldmap;
init.masksignal_init = initParams.masksignal_init;
init.water_r_init = real(initParams.rhoW_init);
init.fat_r_init = real(initParams.rhoF_init);
init.water_i_init = imag(initParams.rhoW_init);
init.fat_i_init = imag(initParams.rhoF_init);

% --- Call the MEX Gateway Function ---
% The name of the MEX function must match the C++ filename.
[wat_r, wat_i, fat_r, fat_i, r2starmap, fm] = fwFit_VarproLS_1r2star_mex(...
    imData, algo, init);

% --- Reshape and Organize Outputs ---
outParams.water_amp = wat_r + 1i * wat_i;
outParams.fat_amp = fat_r + 1i * fat_i;
outParams.r2starmap = r2starmap;
outParams.fm = fm;

end

