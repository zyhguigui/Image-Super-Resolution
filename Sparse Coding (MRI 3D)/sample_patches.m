function [HP, LP] = sample_patches(image, patch_size, patch_num, upscale)
% Randomly crop high resolution and low resolution patches from 2D or 3D image data
% INPUTS:
%   image: 2D or 3D high resolution image data.
%   patch_size: scalar, patches are of size patch_size*patch_size*patch_size.
%   patch_hum: number of patches cropped from image.
%   upscale: vector of 2 or 3 elements, corresponding to the dimension of image. scale factor for
%            each dimension.
%
% OUTPUTS:
%   HP: high resolution image patches (size: patch_size^3 * patch_num).
%   LP: low resolution image patches  (size: (4 or 9) * patch_size^3 * patch_hum).

%% For 2D image data
if ndims(image) == 2  %#ok<*ISMAT>
    % generate low resolution counter parts
    lIm = imresize(image, round(size(image)./upscale), 'bicubic');
    lIm = imresize(lIm, size(image), 'bicubic');
    [nrow, ncol] = size(image);
    
    x = randperm(nrow-2*patch_size-1) + patch_size;
    y = randperm(ncol-2*patch_size-1) + patch_size;
    
    [X,Y] = meshgrid(x,y);
    
    X = X(:);
    Y = Y(:);
    
    if patch_num < length(X)
        X = X(1:patch_num);
        Y = Y(1:patch_num);
    end
    patch_num = length(X);
    
    % compute the first and second order gradients
    xf1 = [-1,0,1];
    yf1 = [-1,0,1]';
    
    lImG1x = conv2(lIm, xf1,'same');
    lImG1y = conv2(lIm, yf1,'same');
    
    xf2 = [1,0,-2,0,1];
    yf2 = [1,0,-2,0,1]';
    
    lImG2x = conv2(lIm,xf2,'same');
    lImG2y = conv2(lIm,yf2,'same');
    
    % make patches
    HP = zeros(patch_size^2, patch_num);
    LP = zeros(4*patch_size^2, patch_num); 
    for ii = 1:patch_num
        row = X(ii);
        col = Y(ii);
        
        Hpatch = image(row:row+patch_size-1,col:col+patch_size-1);
        
        Lpatch1 = lImG1x(row:row+patch_size-1,col:col+patch_size-1);
        Lpatch2 = lImG1y(row:row+patch_size-1,col:col+patch_size-1);
        Lpatch3 = lImG2x(row:row+patch_size-1,col:col+patch_size-1);
        Lpatch4 = lImG2y(row:row+patch_size-1,col:col+patch_size-1);

        HP(:,ii) = Hpatch(:)-mean(Hpatch(:));
        LP(:,ii) = [Lpatch1(:);Lpatch2(:);Lpatch3(:);Lpatch4(:)];
    end

%% For 3D image data
elseif ndims(image) == 3
%     lIm = single(affine(image, diag([1./upscale,1]),[],0)); % verbose set to 0
%     lIm = single(affine(lIm, diag([upscale,1]),[],0));    % verbose set to 0
    lIm = single(VolumeResize(image, round(size(image)./upscale),'spline'));
    lIm = single(VolumeResize(lIm, size(image),'spline'));
%     newsize = min([size(image);size(lIm)]); % The size of lIm may be different from hIm
%     nrow = newsize(1); 
%     ncol = newsize(2);
%     npage = newsize(3);
    [nrow, ncol, npage] = size(lIm);
    
    x = randperm(nrow-2*patch_size-1) + patch_size;
    y = randperm(ncol-2*patch_size-1) + patch_size;
    z = randperm(npage-2*patch_size-1) + patch_size;
    
    [X,Y,Z] = meshgrid(x,y,z);
    X = uint16(X(:));
    Y = uint16(Y(:));
    Z = uint16(Z(:));
    
    if patch_num < length(X)
        X = X(1:patch_num);
        Y = Y(1:patch_num);
        Z = Z(1:patch_num);
    end
    patch_num = length(X);
    
    % compute the first order gradients
    xf1 = [-1,0,1]; 
    yf1 = [-1,0,1]';
    zf1 = reshape([-1,0,1],1,1,3);
    
    lImG1x = convn(lIm, xf1,'same');
    lImG1y = convn(lIm, yf1,'same');
    lImG1z = convn(lIm, zf1,'same');
    
%     lImG1xy = sqrt(lImG1x_temp.^2 + lImG1y_temp.^2);
%     lImG1yz = sqrt(lImG1y_temp.^2 + lImG1z_temp.^2);
%     lImG1xz = sqrt(lImG1x_temp.^2 + lImG1z_temp.^2); 
    
%     lImG1xy = lImG1xy - min(lImG1xy(:));       % Normalize the gradients
%     lImG1xy = (lImG1xy / max(lImG1xy(:)))*100;
%     lImG1yz = lImG1yz - min(lImG1yz(:));
%     lImG1yz = (lImG1yz / max(lImG1yz(:)))*100;    
%     lImG1xz = lImG1xz - min(lImG1xz(:));
%     lImG1xz = (lImG1xz / max(lImG1xz(:)))*100;      
    
    % compute the second order gradients
    xf2 = [1,0,-2,0,1];
    yf2 = [1,0,-2,0,1]';
    zf2 = reshape([1,0,-2,0,1],1,1,5);
    
    lImG2x = convn(lIm,xf2,'same');
    lImG2y = convn(lIm,yf2,'same');
    lImG2z = convn(lIm,zf2,'same');
    
%     lImG2xy = sqrt(lImG2x_temp.^2 + lImG2y_temp.^2);
%     lImG2yz = sqrt(lImG2y_temp.^2 + lImG2z_temp.^2);
%     lImG2xz = sqrt(lImG2x_temp.^2 + lImG2z_temp.^2); 
    
%     lImG2xy = lImG2xy - min(lImG2xy(:));       % Normalize the gradients
%     lImG2xy = (lImG2xy / max(lImG2xy(:)))*100;
%     lImG2yz = lImG2yz - min(lImG2yz(:));
%     lImG2yz = (lImG2yz / max(lImG2yz(:)))*100;    
%     lImG2xz = lImG2xz - min(lImG2xz(:));
%     lImG2xz = (lImG2xz / max(lImG2xz(:)))*100;      
    
    % make patches
    HP = single(zeros(patch_size^3, patch_num));
    LP = single(zeros(6*patch_size^3, patch_num));
    for ii = 1:patch_num
        row = X(ii);
        col = Y(ii);
        page = Z(ii);
        
        Hpatch = image(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        
        Lpatch1 = lImG1x(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        Lpatch2 = lImG1y(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        Lpatch3 = lImG1z(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        Lpatch4 = lImG2x(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        Lpatch5 = lImG2y(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        Lpatch6 = lImG2z(row:row+patch_size-1, col:col+patch_size-1, page:page+patch_size-1);
        
        HP(:,ii) = single(Hpatch(:)-mean(Hpatch(:)));
        LP(:,ii) = single([Lpatch1(:);Lpatch2(:);Lpatch3(:);Lpatch4(:);Lpatch5(:);Lpatch6(:)]);
    end
    clear *_temp
else
    error('Input image dimension is not correct.\n');
end

end
