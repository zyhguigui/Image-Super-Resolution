function [hIm] = ScSR(lIm, Dh, Dl, upscale, lambda, overlap, patch_size, interpolated)
% Perform image super resolution.
% INPUTS:
%   lIm: low resolution images, currently should only be 3D array.
%   Dh: dictionary (?)
%   Dl: dictionary (?)
%   upscale: upscale factor in each dimension
%   lambda: sparsity regularization
%   overlap: overlap when extracting patches. The more overlap the better.
%   patch_size: size of patches extracting from images.
%   interpolated: whether the low resolution image lIm has been interpolated.
%       value = 1: we will not interpolate lIm here.
%       value = 0: we will interpolated iIm here.
%
% OUTPUT:
%   hIm: high resolution images
%

% normalize the dictionary
norm_Dl = sqrt(sum(Dl.^2, 1));
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);
Dl(isnan(Dl))=0;   % add on 2017/01/05

% Interpolation of the low-resolution image
if interpolated == 0
    fprintf('    Interpolated low resolution images before super-resolution...');
    lIm = single(affine(lIm, diag([upscale,1]), [], 0));
    fprintf('Done!\n');
end

% Initialize high resolution image
hIm = single(zeros(size(lIm)));
cntMat = single(zeros(size(lIm)));

[row, column, page] = size(lIm);

% extract low-resolution image features
lImfea = extr_lIm_fea(lIm);

% patch indexes for sparse recovery (avoid boundary) 作者为何把row和column调换顺序？
% gridx = 3:patch_size - overlap : column-patch_size-2;
% gridx = [gridx, column-patch_size-2];
% gridy = 3:patch_size - overlap : row-patch_size-2;
% gridy = [gridy, row-patch_size-2];
gridx = [3 : patch_size - overlap : row-patch_size-2, row-patch_size-2];  % 最后一个数为什么要重复一次？
gridy = [3 : patch_size - overlap : column-patch_size-2, column-patch_size-2];
gridz = [3 : patch_size - overlap : page-patch_size-2, page-patch_size-2];

A = Dl'*Dl;
cnt = 0;

% loop to recover each low-resolution patch
for i = 1:length(gridx)
    for j = 1:length(gridy)
        tic;
        for k = 1:length(gridz)
            cnt = cnt+1;
            x = gridx(i);
            y = gridy(j);
            z = gridz(k);
            
            % mPatch = lIm(y:y+patch_size-1, x:x+patch_size-1); % 作者为何把x和y调换顺序？
            mPatch = lIm(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1);
            mMean = mean(mPatch(:));
            mPatch = mPatch(:) - mMean;
            mNorm = sqrt(sum(mPatch.^2));
            
            % mPatchFea = lImfea(y:y+patch_size-1, x:x+patch_size-1, :);
            mPatchFea = lImfea(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1, :);
            mPatchFea = mPatchFea(:);
            mfNorm = sqrt(sum(mPatchFea.^2));
            
            if mfNorm > 1
                m = mPatchFea./mfNorm;
            else
                m = mPatchFea;
            end
            
            % b = -Dl'*m; % original code, slow
            b = -(m' * Dl)'; % modified 2017/01/05
            
            % sparse recovery
            w = L1QP_FeatureSign_yang(lambda, A, b);
            
            % generate the high resolution patch and scale the contrast
            hPatch = Dh * w;
            hPatch = lin_scale(hPatch, mNorm);
            
            hPatch = reshape(hPatch, [patch_size, patch_size, patch_size]);
            hPatch = hPatch + mMean;
            
            % 作者为何把x和y调换顺序？
            % hIm(y:y+patch_size-1, x:x+patch_size-1) = hIm(y:y+patch_size-1, x:x+patch_size-1) + hPatch;
            hIm(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1) = ...
                hIm(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1) + hPatch;
            % cntMat(y:y+patch_size-1, x:x+patch_size-1) = cntMat(y:y+patch_size-1, x:x+patch_size-1) + 1;
            cntMat(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1) = ...
                cntMat(x:x+patch_size-1, y:y+patch_size-1, z:z+patch_size-1) + 1;
        end
        usedtime = toc;
        fprintf('    Slice %d/%d, %d/%d, took %.3fs.\n',i,length(gridx),j,length(gridy),usedtime);
    end
end

% fill in the empty with bicubic interpolation
idx = (cntMat < 1);
hIm(idx) = lIm(idx);

cntMat(idx) = 1;
hIm = hIm./cntMat;
% hIm = uint8(hIm);
end

%% Nested function extr_lIm_dea()
function [lIm_Feature] = extr_lIm_fea( lIm )

if ndims(lIm) == 2  %#ok<*ISMAT>
    lIm_Feature = single(zeros([size(lIm), 4]));
    
    % first order gradient filters
    hf1 = [-1,0,1];
    vf1 = [-1,0,1]';
    
    lIm_Feature(:, :, 1) = conv2(lIm, hf1, 'same');
    lIm_Feature(:, :, 2) = conv2(lIm, vf1, 'same');
    
    % second order gradient filters
    hf2 = [1,0,-2,0,1];
    vf2 = [1,0,-2,0,1]';
    
    lIm_Feature(:, :, 3) = conv2(lIm,hf2,'same');
    lIm_Feature(:, :, 4) = conv2(lIm,vf2,'same');
    
elseif ndims(lIm) == 3
    lIm_Feature = single(zeros([size(lIm), 6]));
    
    % compute the first order gradients
    xf1 = [-1,0,1];
    yf1 = [-1,0,1]';
    zf1 = reshape([-1,0,1],1,1,3);
    
    lImG1x_temp = convn(lIm, xf1,'same');
    lImG1y_temp = convn(lIm, yf1,'same');
    lImG1z_temp = convn(lIm, zf1,'same');
    
    lIm_Feature(:, :, :, 1) = sqrt(lImG1x_temp.^2 + lImG1y_temp.^2);
    lIm_Feature(:, :, :, 2) = sqrt(lImG1y_temp.^2 + lImG1z_temp.^2);
    lIm_Feature(:, :, :, 3) = sqrt(lImG1x_temp.^2 + lImG1z_temp.^2);
    
    % compute the second order gradients
    xf2 = [1,0,-2,0,1];
    yf2 = [1,0,-2,0,1]';
    zf2 = reshape([1,0,-2,0,1],1,1,5);
    
    lImG2x_temp = convn(lIm,xf2,'same');
    lImG2y_temp = convn(lIm,yf2,'same');
    lImG2z_temp = convn(lIm,zf2,'same');
    
    lIm_Feature(:, :, :, 4) = sqrt(lImG2x_temp.^2 + lImG2y_temp.^2);
    lIm_Feature(:, :, :, 5) = sqrt(lImG2y_temp.^2 + lImG2z_temp.^2);
    lIm_Feature(:, :, :, 6) = sqrt(lImG2x_temp.^2 + lImG2z_temp.^2);
end
end

%% Nested function lin_scale()
function [xh] = lin_scale( xh, mNorm )

hNorm = sqrt(sum(xh.^2));

if hNorm
    s = mNorm*1.2/hNorm;
    xh = xh.*s;
end
end