function [Xh, Xl] = rnd_smp_patch(img_path, type, patch_size, num_patch, upscale)

img_dir = dir(fullfile(img_path, type));

img_num = length(img_dir);
nper_img = zeros(1, img_num);

for ii = 1:length(img_dir)
    im = imread(fullfile(img_path, img_dir(ii).name));
    nper_img(ii) = numel(im);
end

nper_img = floor(nper_img*num_patch/sum(nper_img));

% Xh = []; % original code
% Xl = []; % original code
Xh = zeros(patch_size^2, sum(nper_img));   % Add on 2017/01/01
Xl = zeros(4*patch_size^2, sum(nper_img)); % Add on 2017/01/01

for ii = 1:img_num
    patch_num = nper_img(ii);
    im = imread(fullfile(img_path, img_dir(ii).name));
    
    % for MRI images, normalize im between 0 and 255
    if ndims(im) == 2
        im = double(im - min(im(:)));
        im = im / max(im(:)) * 255;
    end
    
    [H, L] = sample_patches(im, patch_size, patch_num, upscale);
    % Xh = [Xh, H]; % original code
    % Xl = [Xl, L]; % original code
    Xh(:, sum(nper_img(1:ii-1))+1:sum(nper_img(1:ii)) ) = H; % Add on 2017/01/01
    Xl(:, sum(nper_img(1:ii-1))+1:sum(nper_img(1:ii)) ) = L; % Add on 2017/01/01
end

% patch_path = ['Training/rnd_patches_' num2str(patch_size) '_' num2str(num_patch) '_s' num2str(upscale) '.mat'];
% save(patch_path, 'Xh', 'Xl');
end