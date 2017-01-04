%% Load and Normalize MRI data
clear;
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

%% Set parameters
dict_size  = 512;          % dictionary size
lambda     = 0.02;         % sparsity regularization, original value 0.15
patch_size = 10;            % image patch size
patch_num  = 10000;        % number of patches to sample, original value 100000
% upscale    = MRI_patient.hdr.dime.pixdim(2:4)./MRI_template.hdr.dime.pixdim(2:4); 
upscale = [1,1,10];         % upscaling factor of three dimensions
upscale(upscale<1) = 1;    % Doing super-resolution, do not downsample the images.

%% Generate image patches
fprintf('Generating image patches...');
% randomly sample image patches
[Xh, Xl] = sample_patches(MRI_template.img, patch_size, patch_num, upscale);

% prune patches with small variances, threshould chosen based on the training data
[Xh, Xl] = patch_pruning(Xh, Xl, 10); % original threshold: 10
fprintf('Done!\n'); 

%% Train dictionaries
fprintf('Training dictionaries...\n');
% joint sparse coding 
[Dh, Dl] = train_coupled_dict(Xh, Xl, dict_size, lambda);

% dict_path = ['Dictionary\\Dict_size' num2str(dict_size) '_sparsity' num2str(lambda) '_scale' regexprep(num2str(upscale),'  ',',') '.mat' ];
dict_path = ['Dictionary\Dict_', datestr(now, 30), '.mat' ];
save(dict_path, 'Dh', 'Dl', 'dict_size', 'lambda', 'patch_size', 'patch_num', 'upscale');
fprintf('Final dictionaries saved in %s.\n\n',dict_path);

%% Recover high resolution images
% =========== testing codes ==============%
% hIm_test = MRI_template.img; %(1:round(end/4),1:round(end/10),1:round(end/10));
% lIm_test = single(affine(hIm_test, diag([1./upscale,1]), [], 0));
% 
% overlap = 4;
% maxIter = 20;
% 
% hIm_re_test = ScSR(lIm_test, Dh, Dl, upscale, lambda, overlap, patch_size, 0); % need to interpolate
% 
% size = min([size(hIm_test);size(hIm_re_test)]);
% hIm_re_test = hIm_re_test(1:size(1),1:size(2),1:size(3));

% hIm_re_test = backprojection(hIm_re_test, lIm_test, maxIter);

