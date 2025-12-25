function outParams = fwFit_MixedLS_1r2star(imDataParams, algoParams, initParams)
%FWFIT_MIXEDLS_1R2STAR MATLAB wrapper for the C++ MixedLS fitting library.
%   Calls the compiled MEX function fwFit_MixedLS_1r2star_mex.
%
%   outParams = fwFit_MixedLS_1r2star(imDataParams, algoParams, initParams)
%
%   INPUTS:
%       imDataParams - Struct with image data and acquisition parameters.
%           .images: [nx, ny, nTE] complex double array. (Column-major)
%           .TE: [1, nTE] double array of echo times in seconds.
%           .FieldStrength: Scalar, in Tesla.
%           .PrecessionIsClockwise: Scalar, typically 1 or -1.
%
%       algoParams - Struct with algorithm settings.
%           .species(1).name = 'water', .frequency = [0], .relAmps = [1]
%           .species(2).name = 'fat', .frequency = [f1..fp], .relAmps = [a1..ap]
%
%       initParams - Struct with initial guesses for the fit.
%           .water_init: [nx, ny] complex double array (water guess).
%           .fat_init: [nx, ny] complex double array (fat guess).
%           .r2s_init: [nx, ny] double array (R2* guess in s^-1).
%           .fm_init: [nx, ny] double array (field map guess in Hz).
%           .masksignal_init: [nx, ny] double array (binary mask, >0.1 is processed).
%
%   OUTPUTS:
%       outParams - Struct containing the fitted parameter maps.
%           .water_amp: [nx, ny] complex double array.
%           .fat_amp: [nx, ny] complex double array.
%           .r2starmap: [nx, ny] double array (in s^-1).
%           .fm: [nx, ny] double array (in Hz).
%           .fit_img: [nx, ny, nTE] complex double array (fitted signal).

% --- Input Validation ---
fields_imData = {'images', 'TE', 'FieldStrength', 'PrecessionIsClockwise'};
fields_algo = {'species'};
fields_init = {'water_init', 'fat_init', 'r2s_init', 'fm_init', 'masksignal_init'};

for i = 1:length(fields_imData)
    if ~isfield(imDataParams, fields_imData{i})
        error('imDataParams missing field: %s', fields_imData{i});
    end
end
if ~isfield(algoParams, fields_algo{1}) || length(algoParams.species) < 2
    error('algoParams must contain .species field with at least 2 elements (water, fat).');
end
for i = 1:length(fields_init)
     if ~isfield(initParams, fields_init{i})
        error('initParams missing field: %s', fields_init{i});
    end
end

% Dimension checks
s_img = size(imDataParams.images);
nx = s_img(1);
ny = s_img(2);
nte = s_img(3);
if length(imDataParams.TE) ~= nte
    error('Dimension mismatch: length(imDataParams.TE) ~= size(imDataParams.images, 3)');
end
s_init = size(initParams.fm_init);
if s_init(1) ~= nx || s_init(2) ~= ny
    error('Dimension mismatch: size(initParams.fm_init) ~= size(imDataParams.images, [1 2])');
end
% Add more checks for other initParams fields if needed

% --- Prepare structs for MEX function ---
imData.images_r = real(imDataParams.images);
imData.images_i = imag(imDataParams.images);
imData.TE = imDataParams.TE(:)'; % Ensure TE is a row vector for memcpy
imData.FieldStrength = imDataParams.FieldStrength;
imData.PrecessionIsClockwise = imDataParams.PrecessionIsClockwise;

algo.species_fat_amp = algoParams.species(2).relAmps(:)'; % Ensure row vector
algo.species_fat_freq = algoParams.species(2).frequency(:)'; % Ensure row vector

init.water_r_init = real(initParams.water_init);
init.water_i_init = imag(initParams.water_init);
init.fat_r_init = real(initParams.fat_init);
init.fat_i_init = imag(initParams.fat_init);
init.r2s_init = initParams.r2s_init;
init.fm_init = initParams.fm_init;
init.masksignal_init = initParams.masksignal_init;

% --- Call the MEX Gateway Function ---
disp('Calling fwFit_MixedLS_1r2star_mex...');
try
    [wat_r, wat_i, fat_r, fat_i, r2starmap, fm, fit_r, fit_i] = fwFit_MixedLS_1r2star_mex(...
        imData, algo, init);
    disp('MEX call successful.');
catch ME
    disp('Error during MEX call:');
    rethrow(ME);
end

% --- Reshape and Organize Outputs ---
% Check if outputs are empty (can happen if MEX fails silently before allocation)
if isempty(wat_r) || isempty(fit_r)
     error('MEX function returned empty arrays. Check for errors during C++ execution.');
end

outParams.water_amp = wat_r + 1i * wat_i;
outParams.fat_amp = fat_r + 1i * fat_i;
outParams.r2starmap = r2starmap;
outParams.fm = fm;

% Reshape the fitted signal - MATLAB stores 3D arrays column-major:
% Element (r,c,t) is at index r + (c-1)*nx + (t-1)*nx*ny (1-based index)
% C++ fills flat array fit_r[idx + t*nx*ny] where idx = r + c*nx (0-based)
% The direct mapping from C++ pointer to MATLAB 3D array works correctly.
outParams.fit_img = fit_r + 1i * fit_i;

disp('Output reshaping complete.');

end

