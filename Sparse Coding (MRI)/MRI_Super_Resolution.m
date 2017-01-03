% Please run MRI_Train_Dictionary.m first if you don't have a dictionary yet.

%% Load MRI data and dictionaries
% After loading dictionaries, we will also have dict_size, lambda, patch_hum, patch_size, upscale
MRI_patient = load_untouch_nii('H:\MRI Data\Cardiac Data\short axis 3D\time_1\ShortAxis3D.nii');
load('Dictionary/Dict_size512_sparsity0.05_scale1,1,2.mat');

%% Set Parameters
lambda = 0.05;                  % sparsity regularization
overlap = 4;                    % the more overlap the better (patch size 5x5)
% up_scale = 2;                 % scaling factor, depending on the trained dictionary
maxIter = 20;                   % if 0, do not use backprojection

%%
% image super-resolution simply basedon interpolation
Img_h_inter = single(affine(MRI_patient.img, upscale, [], 0)); % suppress verbose

% image super-resolution based on sparse representation
[im_h_y] = ScSR(MRI_patient.img, upscae, Dh, Dl, lambda, overlap, patch_size);
[im_h_y] = backprojection(im_h_y, im_l_y, maxIter);

% upscale the chrominance simply by "bicubic" 
[nrow, ncol] = size(im_h_y);
im_h_cb = imresize(im_l_cb, [nrow, ncol], 'bicubic');
im_h_cr = imresize(im_l_cr, [nrow, ncol], 'bicubic');

im_h_ycbcr = zeros([nrow, ncol, 3]);
im_h_ycbcr(:, :, 1) = im_h_y;
im_h_ycbcr(:, :, 2) = im_h_cb;
im_h_ycbcr(:, :, 3) = im_h_cr;
im_h = ycbcr2rgb(uint8(im_h_ycbcr));

% bicubic interpolation for reference
im_b = imresize(im_l, [nrow, ncol], 'bicubic');

% read ground truth image
im = imread('Data/Testing/gnd.bmp');

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im, im_b);
sp_rmse = compute_rmse(im, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);

% show the images
figure, imshow(im_h);
title('Sparse Recovery');
figure, imshow(im_b);
title('Bicubic Interpolation');