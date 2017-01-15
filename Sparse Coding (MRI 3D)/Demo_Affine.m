% This demo shows that for MRI image downsampled by a small factor, a simple trilinear interpolation
% can restore the image to nearly as good as the original one, because the original MRI image itself
% is quite blurry.

%% MRI image
MRI_template = load_untouch_nii('H:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model.nii');
slice_index = 30; % switch which slice to display

% original high resolution image
MRI_Img_h = single(MRI_template.img);
figure; imshow(squeeze(MRI_Img_h(slice_index,:,:)),'DisplayRange',[]); title('original MRI image');

% Low resolution image 1
MRI_Img_l1 = single(affine(MRI_Img_h, diag([1,1,0.5,1]), [], 0));
figure; imshow(squeeze(MRI_Img_l1(slice_index,:,:)),'DisplayRange',[]); title('low resolution 1');

% Low resolution image 2
MIR_Img_l2 = single(affine(MRI_Img_h, diag([1,0.5,0.5,1]), [], 0));
figure; imshow(squeeze(MIR_Img_l2(slice_index,:,:)),'DisplayRange',[]); title('low resolution 2');

% Recovered high resolution image 1 from low resolution image 1 by trilinear interpolation
MIR_Img_h1 = single(affine(MRI_Img_l1, diag([1,1,2,1]), [], 0));
figure; imshow(squeeze(MIR_Img_h1(slice_index,:,:)),'DisplayRange',[]); title('recovery1');

% Recovered high resolution image 2 from low resolution image 2 by trilinear interpolation
MRI_Img_h2 = single(affine(MIR_Img_l2, diag([1,2,2,1]), [], 0));
figure; imshow(squeeze(MRI_Img_h2(slice_index,:,:)),'DisplayRange',[]); title('recovery2');

%% However this is not the case for camera image
Cam_Img_h = imread('cameraman.tif');
figure; imshow(Cam_Img_h); title('original camera image');

Cam_Img_l = imresize(Cam_Img_h, 0.5, 'bilinear');
figure; imshow(Cam_Img_l); title('low resolution camera image');

Cam_Img_h_re = imresize(Cam_Img_l, 2, 'bilinear');
figure; imshow(Cam_Img_h_re); title('recovered camera image');