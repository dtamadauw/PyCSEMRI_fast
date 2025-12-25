%function ff = computeFF( outParams )

function ff = computeFF( outParams, ffParams )

if nargin<2
  ffParams.noise_bias_correction = 1;
end

        
curf = outParams.fat_amp;
curw = outParams.water_amp;


denom = (abs(curf) + abs(curw));
denom2 = denom;
denom2(denom==0) = 1; % To avoid divide-by-zero issues
ff = 100*abs(curf)./denom2;


if ffParams.noise_bias_correction>0
  fatregions = ff>50;
  watregions = ff<=50;
  denom2 = abs(curf + curw);
  %denom2(denom==0) = 1; % To avoid divide-by-zero issues
  denom2(abs(denom)<0.01) = 1;
  ff(watregions) = 100 - 100*abs(curw(watregions))./denom2(watregions);
  ff(fatregions) = 100*abs(curf(fatregions))./denom2(fatregions);
end

% curf = outParams.species(2).amps;
% curw = outParams.species(1).amps;

% ff1 = 100*abs(curf)./(abs(curf) + abs(curw));
% 
% fatregions = ff1>50;
% watregions = ff1<=50;
% all_ff = 100*abs(curf)./(abs(curf) + abs(curw));
% all_ff(watregions) = 100 - 100*abs(curw(watregions))./abs(curf(watregions) + curw(watregions));
% all_ff(fatregions) = 100*abs(curf(fatregions))./abs(curf(fatregions) + curw(fatregions));
% 
% 
% ff = all_ff;
% ff(ff > 500) = nan;
% ff(ff < -500) = nan;