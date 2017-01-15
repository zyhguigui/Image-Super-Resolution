function [hIm] = ScSR(lIm, up_scale, Dh, Dl, lambda, overlap)

% normalize the dictionary
norm_Dl = sqrt(sum(Dl.^2, 1)); 
Dl = Dl./repmat(norm_Dl, size(Dl, 1), 1);
Dl(isnan(Dl))=0;   % add on 2017/01/05

patch_size = sqrt(size(Dh, 1));

% bicubic interpolation of the low-resolution image
mIm = single(imresize(lIm, up_scale, 'bicubic'));

hIm = zeros(size(mIm));
cntMat = zeros(size(mIm));

[h, w] = size(mIm);

% extract low-resolution image features
lImfea = extr_lIm_fea(mIm);

% patch indexes for sparse recovery (avoid boundary)
gridx = 3:patch_size - overlap : w-patch_size-2;
gridx = [gridx, w-patch_size-2];
gridy = 3:patch_size - overlap : h-patch_size-2;
gridy = [gridy, h-patch_size-2];

A = Dl'*Dl;
cnt = 0;

% loop to recover each low-resolution patch
for ii = 1:length(gridx)
    tic;
    sparsity = 0;
    for jj = 1:length(gridy)
        cnt = cnt+1;
        xx = gridx(ii);
        yy = gridy(jj);
        
        mPatch = mIm(yy:yy+patch_size-1, xx:xx+patch_size-1);
        mMean = mean(mPatch(:));
        mPatch = mPatch(:) - mMean;
        mNorm = sqrt(sum(mPatch.^2));
        
        mPatchFea = lImfea(yy:yy+patch_size-1, xx:xx+patch_size-1, :);   
        mPatchFea = mPatchFea(:);
        mfNorm = sqrt(sum(mPatchFea.^2));
        
        if mfNorm > 1
            y = mPatchFea./mfNorm;
        else
            y = mPatchFea;
        end
        
        b = -Dl'*y;
      
        % sparse recovery
        w = L1QP_FeatureSign_yang(lambda, A, b);
        sparsity = sparsity + sum(w~=0)/length(w);
        
        % generate the high resolution patch and scale the contrast
        hPatch = Dh*w;
        hPatch = lin_scale(hPatch, mNorm);
        
        hPatch = reshape(hPatch, [patch_size, patch_size]);
        hPatch = hPatch + mMean;
        
        hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) = hIm(yy:yy+patch_size-1, xx:xx+patch_size-1) + hPatch;
        cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) = cntMat(yy:yy+patch_size-1, xx:xx+patch_size-1) + 1;
    end
    sparsity = sparsity / jj;
    usedtime = toc;
    fprintf('Slice %d/%d, sparsity = %f, took %.3fs.\n', ii, length(gridx), sparsity, usedtime);
end

% fill in the empty with bicubic interpolation
idx = (cntMat < 1);
hIm(idx) = mIm(idx);

cntMat(idx) = 1;
hIm = hIm./cntMat;
hIm = hIm - min(hIm(:));
% hIm = uint16(hIm);
end

%% Nested function extr_lIm_fea()
function [lImFea] = extr_lIm_fea( lIm )

[nrow, ncol] = size(lIm);

lImFea = zeros([nrow, ncol, 4]);

% first order gradient filters
hf1 = [-1,0,1];
vf1 = [-1,0,1]';
 
lImFea(:, :, 1) = conv2(lIm, hf1, 'same');
lImFea(:, :, 2) = conv2(lIm, vf1, 'same');

% second order gradient filters
hf2 = [1,0,-2,0,1];
vf2 = [1,0,-2,0,1]';
 
lImFea(:, :, 3) = conv2(lIm,hf2,'same');
lImFea(:, :, 4) = conv2(lIm,vf2,'same');
end

%% Nested function lin_scale()
function [xh] = lin_scale( xh, mNorm )

hNorm = sqrt(sum(xh.^2));

if hNorm
    s = mNorm*1.2/hNorm;
    xh = xh.*s;
end
end