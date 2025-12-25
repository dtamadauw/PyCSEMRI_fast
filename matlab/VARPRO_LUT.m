function [initParams] = VARPRO_LUT(imDataParams, algoParams)

[fm_init, r2s_init, mask, wat_r, fat_r, wat_i, fat_i] = VARPRO_LUT_mex(imDataParams, algoParams);

rhoW = complex(wat_r,wat_i);
rhoF = complex(fat_r,fat_i);

    initParams.r2starmap = r2s_init; % Flat initial guess
    initParams.fieldmap = fm_init; % Good guess for fieldmap
    initParams.masksignal_init = mask;
    initParams.rhoW_init = rhoW;
    initParams.rhoF_init = rhoF;

    species(1).name = algoParams.species(1).name;
    species(2).name = algoParams.species(2).name;

    species(1).amps = rhoW;
    species(2).amps = rhoF;

    initParams.species = species;

