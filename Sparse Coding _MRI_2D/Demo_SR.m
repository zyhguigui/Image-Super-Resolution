% =========================================================================
% Simple demo codes for image super-resolution via sparse representation
%
% Reference
%   J. Yang et al. Image super-resolution as sparse representation of raw
%   image patches. CVPR 2008.
%   J. Yang et al. Image super-resolution via sparse representation. IEEE 
%   Transactions on Image Processing, Vol 19, Issue 11, pp2861-2873, 2010
%
% Jianchao Yang
% ECE Department, University of Illinois at Urbana-Champaign
% For any questions, send email to jyang29@uiuc.edu
% =========================================================================

% clear all; clc;
% read ground truth image
im_gnd = imread('Data/Testing/gnd.bmp');
size_gnd = size(im_gnd);
% im_gnd_ycbcr = rgb2ycbcr(im_gnd);
% im_gnd_y = im_gnd_ycbcr(:,:,1);

upscale = [2,2];                   % scaling factor, depending on the trained dictionary

% read test image
im_l = imresize(im_gnd, round(size_gnd(1:2)./upscale), 'bicubic');
% im_l = imread('Data/Testing/input.bmp');

% load dictionary
load('Dictionary/Dict_20170109T110607.mat');
% load('Dictionary/D_512_0.15_5.mat');

% set parameters
lambda = 0.2;                   % sparsity regularization
overlap = 4;                    % the more overlap the better (patch size 5x5)
maxIter = 20;                   % if 0, do not use backprojection

% change color space, work on illuminance only
im_l_ycbcr = rgb2ycbcr(im_l);
im_l_y = im_l_ycbcr(:, :, 1);
im_l_cb = im_l_ycbcr(:, :, 2);
im_l_cr = im_l_ycbcr(:, :, 3);

% image super-resolution based on sparse representation
[im_h_y] = ScSR(im_l_y, size_gnd(1:2), Dh, Dl, lambda, overlap);
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



% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im_gnd, im_b);
sp_rmse = compute_rmse(im_gnd, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);

% show the images
figure, imshow(im_h);
title('Sparse Recovery');
figure, imshow(im_b);
title('Bicubic Interpolation');