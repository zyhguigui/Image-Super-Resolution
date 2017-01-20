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
im_gnd = imread('Data/MRIvalidation/valid3.png');

% normalize gnd truth image
im_gnd = double(im_gnd - min(im_gnd(:)));
im_gnd = im_gnd / max(im_gnd(:)) * 255;

size_gnd = size(im_gnd);

% scaling factor, depending on the trained dictionary
upscale = [1,10];

% load dictionary
load('Dictionary/Dict_20170112T133007.mat');

% set parameters
lambda = 0.2;                   % sparsity regularization
overlap = 9;                    % the more overlap the better (patch size 5x5)
maxIter = 20;                   % if 0, do not use backprojection

% read test image
im_l = imresize(im_gnd, round(size_gnd(1:2)./upscale), 'bicubic');

% change color space, work on illuminance only
% im_l_ycbcr = rgb2ycbcr(im_l);
% im_l_y = im_l_ycbcr(:, :, 1);
% im_l_cb = im_l_ycbcr(:, :, 2);
% im_l_cr = im_l_ycbcr(:, :, 3);

% image super-resolution based on sparse representation
[im_h] = ScSR(im_l, size_gnd(1:2), Dh, Dl, lambda, overlap);
[im_h] = backprojection(im_h, im_l, maxIter);

% normalize im_h
im_h = double(im_h - min(im_h(:)));
im_h = im_h / max(im_h(:)) * 255;

% upscale the chrominance simply by "bicubic" 
[nrow, ncol] = size(im_h);
im_b = imresize(im_l, [nrow, ncol], 'bicubic');

% normalize im_b
im_b = double(im_b - min(im_b(:)));
im_b = im_b / max(im_b(:)) * 255;

% compute PSNR for the illuminance channel
bb_rmse = compute_rmse(im_gnd, im_b);
sp_rmse = compute_rmse(im_gnd, im_h);

bb_psnr = 20*log10(255/bb_rmse);
sp_psnr = 20*log10(255/sp_rmse);

fprintf('PSNR for Bicubic Interpolation: %f dB\n', bb_psnr);
fprintf('PSNR for Sparse Representation Recovery: %f dB\n', sp_psnr);

% show the images
figure; 
subplot(1,3,1); imshow(im_h,'DisplayRange',[]); title('reconstructed');
subplot(1,3,2); imshow(im_b,'DisplayRange',[]); title('bicubic');
subplot(1,3,3); imshow(im_gnd,'DisplayRange',[]); title('original');


