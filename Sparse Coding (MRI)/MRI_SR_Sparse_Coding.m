%% Load MRI data
clear;
fprintf('Loading data...');
MRI_template = load_untouch_nii('H:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model.nii');
MRI_patient = load_untouch_nii('H:\MRI Data\Cardiac Data\short axis 3D\time_1\ShortAxis3D.nii');
MRI_template.img = single(MRI_template.img) / single(max(MRI_template.img(:))) * 100; % Normalize data
MRI_patient.img = single(MRI_patient.img) / single(max(MRI_patient.img(:))) * 100; % Normalize data
fprintf('Done!\n');

%% Set parameters
dict_size  = 512;          % dictionary size
lambda     = 0.15;         % sparsity regularization
patch_size = 5;            % image patch size
patch_num  = 100000;       % number of patches to sample
% upscale    = MRI_patient.hdr.dime.pixdim(2:4)./MRI_template.hdr.dime.pixdim(2:4); % upscaling factor of three dimensions
upscale = [1,1,2];
upscale(upscale<1) = 1;    % Doing super-resolution, do not downsample the images.
% upscale(upscale>2) = 2;

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
[Dh, Dl] = train_coupled_dict(Xh, Xl, dict_size, lambda, upscale);
dict_path = ['Dictionary/D_' num2str(dict_size) '_' num2str(lambda) '_' num2str(patch_size) '.mat' ];
save(dict_path, 'Dh', 'Dl');