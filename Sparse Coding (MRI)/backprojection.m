function [im_h] = backprojection(im_h, im_l, maxIter)

% [row_l, col_l] = size(im_l); % original code
% [row_h, col_h] = size(im_h); % original code
scale = size(im_l) ./ size(im_h);
size_im_h = size(im_h);

% original codes
% p = fspecial('gaussian', 5, 1);
% p = p.^2;
% p = p./sum(p(:));

% Comment on 2017/01/03
% im_l = double(im_l);
% im_h = double(im_h);

for i = 1:maxIter
    % im_l_s = imresize(im_h, [row_l, col_l], 'bicubic'); % original code
    im_l_s = single(affine(im_h, diag([scale,1]), [], 0)); % turnoff verbose
    im_diff = im_l - im_l_s;
    
    % im_diff = imresize(im_diff, [row_h, col_h], 'bicubic'); % original code
    im_diff = single(affine(im_diff, diag([1./scale,1]), [], 0)); % turnoff verbose
    size_im_diff = size(im_diff);
    
    % Make sure the dimension is consistant
    if min(size_im_h./size_im_diff) < 1
        im_diff = im_diff(1:size_im_h(1),1:size_im_h(2),1:size_im_h(3));
        size_im_diff = size(im_diff);
    end
    if max(size_im_h./size_im_diff) > 1
        size_diff = size_im_h - size_im_diff;
        for ii = 1:length(size_diff)
           if size_diff(ii) == 0
               continue;
           else
               im_diff = cat(ii, im_diff, im_diff()); % Not done here!!!!!!!!!!!
           end
           
        end
    end
    
    % im_h = im_h + conv2(im_diff, p, 'same'); % original code
    im_h = im_h + imgaussfilt3(im_diff, 1, 'FilterSize', 5);  % 2017/01/03: filter by 5x5x5 gaussian filter with std = 1
end
end