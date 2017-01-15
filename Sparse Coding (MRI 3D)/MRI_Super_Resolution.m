%% Load and Normalize MRI data
fprintf('Loading data...');
MRI_template = load_untouch_nii('D:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model_truncate.nii');
% MRI_patient = load_untouch_nii('D:\MRI Data\Cardiac Data\short axis 3D\time_1\ShortAxis3D.nii');
fprintf('Done!\n');

fprintf('Normalizing data...') % Normalize data to range 0~100
MRI_template.img = single(MRI_template.img - min(MRI_template.img(:)));
MRI_template.img = MRI_template.img / max(MRI_template.img(:)) * 255; 
% MRI_patient.img = single(MRI_patient.img - min(MRI_patient.img(:)));
% MRI_patient.img = MRI_patient.img / max(MRI_patient.img(:)) * 100;
fprintf('Done!\n');

%% Load dictionary
load('Dictionary/Dict_20170115T010241.mat');
overlap = 4;
maxIter = 20;
lambda = 0.05;

%% Recover high resolution images
% =========== testing codes ==============%
% High resolution image
hIm_gnd = MRI_template.img; %(1:round(end/4),1:round(end/10),1:round(end/10));
hIm_gnd_truncate = hIm_gnd(40:50, :, :); % truncate for faster calculation

% low resolution image
lIm_test = single(VolumeResize(hIm_gnd_truncate, round(size(hIm_gnd_truncate)./upscale),'spline'));

% Interpolated "high" resoution image
lIm_Interpolate = single(VolumeResize(lIm_test, size(hIm_gnd_truncate),'spline'));

% recovered high resolution image
fprintf('Performing reconstruction...\n');
hIm_sr_nofinetune = ScSR(lIm_Interpolate, Dh, Dl, lambda, overlap, patch_size, size(hIm_gnd_truncate)); % Don't need to interpolate
minsize = min([size(hIm_gnd_truncate);size(hIm_sr_nofinetune)]);
hIm_sr_nofinetune = hIm_sr_nofinetune(1:minsize(1),1:minsize(2),1:minsize(3));
fprintf('Reconstruction finished!\n\n');

% fine-tuning of recovered high resolution image
fprintf('Performing fine-tuning of the reconstructed image...');
hIm_sr_finetune = backprojection(hIm_sr_nofinetune, lIm_test, maxIter);
fprintf('Done!\n\n');

% Display results
slice = 5;
figure; imshow(squeeze(hIm_gnd_truncate(slice,:,:)),'DisplayRange',[]); title('Ground truth image');
figure; imshow(squeeze(lIm_Interpolate(slice,:,:)),'DisplayRange',[]); title('Interpolated low resolution image');
figure; imshow(squeeze(hIm_sr_finetune(slice,:,:)),'DisplayRange',[]); title('Recovered high resolution image');

% Compute PSNR
im_gnd = squeeze(hIm_gnd_truncate(slice,:,:));
im_gnd = im_gnd - min(im_gnd(:));
im_gnd = im_gnd / max(im_gnd(:)) * 255;

im_b = squeeze(lIm_Interpolate(slice,:,:));
im_b = im_b - min(im_b(:));
im_b = im_b / max(im_b(:)) * 255;

im_h = squeeze(hIm_sr_finetune(slice,:,:));
im_h = im_h - min(im_h(:));
im_h = im_h / max(im_h(:)) * 255;

bb_rmse = compute_rmse(im_gnd, im_b);
sp_rmse = compute_rmse(im_gnd, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);
