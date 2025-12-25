% --- Script to run an example reconstruction ---
clear; clc; close all;

%% Define Algorithm Parameters from Specification [cite: 711, 712, 713]
params.f_range_coarse = -300:5:300; % [Hz] [cite: 641]
params.f_range_refine = -5:1:5;     % [Hz] [cite: 642]
params.r2_grid = [20, 30, 45, 70, 100, 150, 250, 400]; % [s^-1] [cite: 643, 684]
params.fat_spec.f_ppm = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]; % Example peaks
params.fat_spec.amp   = [0.087 0.693 0.128 0.004 0.039 0.048]; % Relative amps [cite: 626]
params.larmor_freq_MHz = 63.87; % For 1.5T
params.K = 2;               % Number of candidates to keep per voxel [cite: 651, 685]
params.mu = 0.6;           % Spatial regularization parameter [cite: 663, 691]
params.mask_threshold = 0.05;% Threshold for foreground mask [cite: 637]
params.neighborhood = 8;    % 8-neighbor for 2D spatial coupling [cite: 689]
params.SUBSAMPLE = 4;
%% Generate Phantom Data
dims = [128, 128];
%h5_name = '/home/dxt005/r-fcb-isilon/groups/ReederGroup/dhernando/work/datasets/288_FAM_Clinical/EXAM_1112_2025_02_13/IDEAL/IdealChanCombData_1112_7.h5';
%h5_name = '/home/dxt005/home/004_ConfidenceMap/data_validation/EXAM_25585_UWMR3_2022_05_19/IDEAL/IdealChanCombData_25585_4.h5';
%h5_name = '/home/dxt005/home/004_ConfidenceMap/data_validation/EXAM_2504_TACMR1_2022_08_02/IDEAL/IdealChanCombData_2504_5.h5';
%h5_name = '/home/dxt005/home/004_ConfidenceMap/data_validation/EXAM_1524_TACMR1_2022_05_27/IDEAL/IdealChanCombData_1524_4.h5';
h5_name = '/home/dxt005/home/016_FAM_Recon/data/Gavin_Nov052025/Exam6478/IdealChanCombData_6478_6.h5';

h5struct = loadh5(h5_name);
hdr = h5struct.Header;
TE_in_ms = 1000*hdr.EchoTimes(1:hdr.NumEchoes).';
useTEs = 1:hdr.NumEchoes; 
TEs = hdr.EchoTimes(useTEs).';
imDataParams.FieldStrength = hdr.FieldStrength/1e4;
nslices = hdr.NumSlices;
myslices = 1:nslices;
clear('ims')
for ksl=1:length(myslices)

    disp([h5_name ', Slice ' num2str(myslices(ksl))]);

    ims(:,:,:,ksl) = eval(['h5struct.Data.Slice' num2str(ksl-1)]);   
end
ims = ims(:,end:-1:1,:,:,:);  
ims = permute(ims,[1 2 4 5 3]);
img_recon = ims./max(abs(ims(:)))*100;
img_recon = squeeze(img_recon);
img_recon = permute(img_recon, [1 2 4 3]);
t_n  = TEs;
params.larmor_freq_MHz = 42.58*imDataParams.FieldStrength; % For 1.5T

img_recon(1,1,:,17) = img_recon(129,128,:,17);

% mask_init = zeros(256,256,size(img_recon,4));
% for ii=1:(size(img_recon,4))
% 
%     img_corrected = Intensity_correction(abs(img_recon(:,:,1,ii)));
%     T = 0.9*median(img_corrected);
%     mask_init(:,:,ii) = img_corrected>T;
% end


%% Run Reconstruction
% --- imDataParams ---
imDataParams.images =(img_recon(:,:,:,17));
imDataParams.TE = TEs;
imDataParams.FieldStrength = imDataParams.FieldStrength; % Tesla
imDataParams.PrecessionIsClockwise = -1;

% --- algoParams ---
algoParams.species(1).name = 'water';
algoParams.species(1).frequency = 0.0;
algoParams.species(1).relAmps = 1.0;
algoParams.species(2).name = 'fat';
algoParams.species(2).frequency = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]; % ppm
algoParams.species(2).relAmps = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];
algoParams.NUM_FMS = 121; % Number of field map values to discretize
algoParams.range_r2star = [10 500]; % Range of R2* values
algoParams.NUM_R2STARS = 11; % Number of R2* values for quantization
algoParams.range_fm = [-250 250]; % Range of field map values
algoParams.mu = 0.01;
algoParams.mask_threshold = 0.05;
algoParams.SUBSAMPLE = 4;

% algoParams.SUBSAMPLE = 4;
%algoParams.range_fm = [-140 300];
%Estimate fm range
% imDataParams.images =(img_recon(:,:,:,32));
% algoParams.mask_threshold = 0.01;
% tic;
% [initParams] = VARPRO_LUT(imDataParams, algoParams);
% toc;

algoParams.range_fm = [ -200 200 ];
algoParams.mask_threshold = 0.005;

fm_init = zeros(256,256,size(img_recon,4));
r2s_init = zeros(256,256,size(img_recon,4));
rhoW_init = zeros(256,256,size(img_recon,4));
rhoF_init = zeros(256,256,size(img_recon,4));
mask_init = zeros(256,256,size(img_recon,4));
for ii=1:(size(img_recon,4))
    tic;
    imDataParams.images =conj(img_recon(:,:,:,ii));
        toc;
end

%%
% params.SUBSAMPLE = 8;
% params.mask_threshold = 0.1;% Threshold for foreground mask [cite: 637]
% 
% tic;
% [results] = fast_ideal_reconstruction(conj(img_recon(:,:,:,7)), t_n, params);
% toc;

%%

images = (img_recon(:,:,:,17));
initParams = [];
initParams.r2s_init = r2s_init(:,:,17); % Flat initial guess
initParams.fm_init = fm_init(:,:,17); % Good guess for fieldmap
initParams.masksignal_init = 1+0*mask_init(:,:,17);
initParams.rhoW_init = rhoW_init(:,:,17);
initParams.rhoF_init = rhoF_init(:,:,17);

% --- imDataParams ---

imDataParams.TE = TEs;
imDataParams.FieldStrength = imDataParams.FieldStrength; % Tesla
imDataParams.PrecessionIsClockwise = -1;

% --- algoParams ---
algoParams.species(1).name = 'water';
algoParams.species(1).frequency = 0.0;
algoParams.species(1).relAmps = 1.0;
algoParams.species(2).name = 'fat';
algoParams.species(2).frequency = [-3.80, -3.40, -2.60, -1.94, -0.39, 0.60]; % ppm
algoParams.species(2).relAmps = [0.087, 0.693, 0.128, 0.004, 0.039, 0.048];


fprintf('Running C++ VARPRO fitting via MATLAB wrapper...\n');
PDFF_3D = zeros(size(img_recon,1),size(img_recon,2),size(img_recon,3));
R2s_3D = zeros(size(img_recon,1),size(img_recon,2),size(img_recon,3));

for ii=1:(size(img_recon,4))
    disp(['Slice: ' num2str(ii)]);
    imDataParams.images = (img_recon(:,:,:,ii));
%     initParams.r2s_init = r2s_init(:,:,ii); % Flat initial guess
%     initParams.fm_init = fm_init(:,:,ii); % Good guess for fieldmap
%     initParams.masksignal_init = 1+0*mask_init(:,:,ii);
%     initParams.rhoW_init = rhoW_init(:,:,ii);
%     initParams.rhoF_init = rhoF_init(:,:,ii);
    initParams = VARPRO_LUT(imDataParams, algoParams);
    fm_init(:,:,ii) = initParams.fieldmap;

    outParams = fwFit_VarproLS_1r2star(imDataParams, algoParams, initParams);
    %outParams = fwFit_MagnLS_1r2star(imDataParams, algoParams, initParams);
    PDFF_3D(:,:,ii) = computeFF(outParams);
    R2s_3D(:,:,ii) = outParams.r2starmap;
end


%rep_dir = '/home/dxt005/home/004_ConfidenceMap/PyCSEMRI_fast/matlab/EX25585/';
%rep_dir = '/home/dxt005/home/004_ConfidenceMap/PyCSEMRI_fast/matlab/EXAM1524/';
%write_dicom_replace(rep_dir, PDFF_3D, 11, 'NewRecon PDFF', rep_dir);

tic;

toc;

tic;
%outParams = fwFit_ComplexLS_1r2star(imDataParams, algoParams, initParams);
toc;

tic;
%outParams = fwFit_MagnLS_1r2star(imDataParams, algoParams, initParams);
toc;

%initParams.water_init = complex(1,0)+0*initParams.fm_init;
%initParams.fat_init = complex(1,0)+0*initParams.fm_init;
tic;
%outParams = fwFit_MixedLS_1r2star(imDataParams, algoParams, initParams);
toc;

%%
fm_3D = [];
for ii=1:size(img_recon,4)
    imDataParams.images = (img_recon(:,:,:,ii));
    %[results] = fast_ideal_reconstruction(conj(img_recon(:,:,:,ii)), t_n, params);
    initParams.r2s_init = r2s_init(:,:,ii); % Flat initial guess
    initParams.fm_init = fm_init(:,:,ii); % Good guess for fieldmap
    initParams.masksignal_init = mask_init(:,:,ii);
    tic;
    outParams = fwFit_VarproLS_1r2star(imDataParams, algoParams, initParams);
    toc;
    fm_3D(:,:,ii) = outParams.fm;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ========================= VISUAL DEBUGGING ========================== %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% 1. Debug Stage 2: Check Voxel-wise Candidate Generation (2D Surface)
fprintf('\n--- Displaying Debugging Plots ---\n');

figure('Name', 'Debug Stage 2: Voxel-wise 2D Residual Surface', 'Position', [50, 50, 1200, 500]);

% Plot the 2D residual surface
subplot(1,2,1);
imagesc(params.f_range_coarse, params.r2_grid, debug_data.candidates.debug_residual_surface);
hold on;
set(gca, 'YDir', 'normal'); % Put R2* in ascending order
colorbar;
title({'2D Residual Surface R(fB, R2*) for single voxel', ['(idx=', num2str(debug_data.candidates.debug_voxel_idx), ')']});
xlabel('Field Map (fB) [Hz]');
ylabel('R2* [s^{-1}]');
colormap(gca, 'hot');

% Overlay the true parameter values
true_fB = debug_data.true_fB_debug_voxel;
true_R2star = debug_data.true_R2star_debug_voxel;
plot(true_fB, true_R2star, 'gx', 'MarkerSize', 15, 'LineWidth', 3);

% Overlay the candidate found by the algorithm
cand_f = debug_data.candidates.f_cand(debug_data.candidates.debug_voxel_idx, 1);
cand_r2 = debug_data.candidates.r2_cand(debug_data.candidates.debug_voxel_idx, 1);
plot(cand_f, cand_r2, 'wo', 'MarkerSize', 15, 'LineWidth', 3);
legend('Residual', 'True Minimum', 'Found Minimum', 'Location', 'northwest');


% Plot the "Best Candidate Map" (b_q), which is the input to Stage 3
subplot(1,2,2);
imagesc(debug_data.b_q_map); axis image off; colorbar;
title('Stage 3 Input: Best Candidate Map (b_q)');
colormap(gca, 'jet');

%% 2. Debug Stage 3: Check Spatial Coupling
figure('Name', 'Debug Stage 3: Spatial Coupling', 'Position', [100, 100, 1500, 400]);
subplot(1,3,1);
imagesc(debug_data.a_q_map); axis image off; colorbar;
title('Stage 3 Input: Curvature Weights (a_q)');

subplot(1,3,2);
imagesc(debug_data.fB_coupled_map); axis image off; colorbar;
title('Stage 3 Output: Spatially Coupled Map');
colormap(gca, 'jet');

subplot(1,3,3);
imagesc(results.fB .* results.mask); axis image off; colorbar;
title('Stage 4 Output: Final Refined Map');
colormap(gca, 'jet');

%% 3. Final Results Comparison
figure('Name', 'Reconstruction Results', 'Position', [150, 150, 1200, 800]);
subplot(1,4,1); imagesc(results.fB .* results.mask); axis image off; colorbar; title('Estimated Field Map (Hz)'); colormap(gca,'jet');
subplot(1,4,2); imagesc(results.PDFF .* results.mask); axis image off; colorbar; caxis([0 1]); title('Estimated PDFF');
subplot(1,4,3); imagesc(results.R2star .* results.mask); axis image off; colorbar; title('Estimated R2* (s^{-1})');
subplot(1,4,4); imagesc(abs(results.rhoW) .* results.mask); axis image off; colorbar; title('Estimated Water');
colormap(gray);
