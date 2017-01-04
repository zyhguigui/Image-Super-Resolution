%% Load and Normalize MRI data
fprintf('Loading data...');
MRI_template = load_untouch_nii('H:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model.nii');
% MRI_patient = load_untouch_nii('H:\MRI Data\Cardiac Data\short axis 3D\time_1\ShortAxis3D.nii');
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
hIm_gnd_trunc = hIm_gnd(40:50, 101:300, 101:300); % truncate for faster calculation

% low resolution image
lIm_test = single(affine(hIm_gnd_trunc, diag([1./upscale,1]), [], 0));

% recovered high resolution image
fprintf('Performing reconstruction...\n');
hIm_re = ScSR(lIm_test, Dh, Dl, upscale, lambda, overlap, patch_size, 0); % need to interpolate

% fine-tuning of recovered high resolution image
size = min([size(hIm_gnd_trunc);size(hIm_re)]);
hIm_re = hIm_re(1:size(1),1:size(2),1:size(3));
hIm_re = backprojection(hIm_re, lIm_test, maxIter);
