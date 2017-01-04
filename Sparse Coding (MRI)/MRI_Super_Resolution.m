%% Load and Normalize MRI data
fprintf('Loading data...');
MRI_template = load_untouch_nii('D:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model.nii');
% MRI_patient = load_untouch_nii('D:\MRI Data\Cardiac Data\short axis 3D\time_1\ShortAxis3D.nii');
fprintf('Done!\n');

fprintf('Normalizing data...') % Normalize data to range 0~100
MRI_template.img = single(MRI_template.img - min(MRI_template.img(:)));
MRI_template.img = MRI_template.img / max(MRI_template.img(:)) * 100; 
% MRI_patient.img = single(MRI_patient.img - min(MRI_patient.img(:)));
% MRI_patient.img = MRI_patient.img / max(MRI_patient.img(:)) * 100;
fprintf('Done!\n');

%% Load dictionary
load('Dictionary/Dict_20170104T105405.mat');
overlap = 4;
maxIter = 20;

%% Recover high resolution images
% =========== testing codes ==============%
% High resolution image
hIm_gnd = MRI_template.img; %(1:round(end/4),1:round(end/10),1:round(end/10));
hIm_gnd_truncate = hIm_gnd(40:50, :, :); % truncate for faster calculation

% low resolution image
lIm_test = single(affine(hIm_gnd_truncate, diag([1./upscale,1]), [], 0));

% Interpolated "high" resoution image
lIm_Interpolate = single(affine(lIm_test, diag([upscale,1]), [], 0));

% recovered high resolution image
fprintf('Performing reconstruction...\n');
hIm_sr_nofinetune = ScSR(lIm_test, Dh, Dl, upscale, lambda, overlap, patch_size, 0); % need to interpolate
size = min([size(hIm_gnd_truncate);size(hIm_sr_nofinetune)]);
hIm_sr_nofinetune = hIm_sr_nofinetune(1:size(1),1:size(2),1:size(3));

% fine-tuning of recovered high resolution image
hIm_sr_finetune = backprojection(hIm_sr_nofinetune, lIm_test, maxIter, upscale);

% Display results
slice = 5;
figure; imshow(squeeze(hIm_gnd_truncate(slice,:,:)),'DisplayRange',[]); title('Ground truth image');
figure; imshow(squeeze(lIm_Interpolate(slice,:,:)),'DisplayRange',[]); title('Interpolated low resolution image');
figure; imshow(squeeze(hIm_sr_finetune(slice,:,:)),'DisplayRange',[]); title('Recovered high resolution image');

