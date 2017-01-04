function [im_h] = backprojection(im_h, im_l, maxIter,upscale)

% [row_l, col_l] = size(im_l); % original code
% [row_h, col_h] = size(im_h); % original code

% add on 2017/01/04
size_im_l = size(im_l);
size_im_h = size(im_h);
% scale = size(im_l) ./ size(im_h);

% original codes
% p = fspecial('gaussian', 5, 1);
% p = p.^2;
% p = p./sum(p(:));

% Comment on 2017/01/03
% im_l = double(im_l);
% im_h = double(im_h);

for i = 1:maxIter
    % Make low resolution image from recovered high resolution image
    im_ls = single(affine(im_h, diag([1./upscale,1]), [], 0)); % turnoff verbose
    
    % Make sure im_l_s and im_l have the same size
    size_im_ls = size(im_ls);
    if min(size_im_l./size_im_ls) < 1
        im_ls = im_ls(1:size_im_l(1),1:size_im_l(2),1:size_im_l(3));
        size_im_ls = size(im_ls);
    end
    if max(size_im_l./size_im_ls) > 1
        diffsize = size_im_l - size_im_ls;
        for ii = 1:length(diffsize)
           if diffsize(ii) == 0
               continue;
           end
           switch ii
               case 1
                   im_ls = cat(1, im_ls, im_ls(end-diffsize(1)+1:end, :, :));
               case 2
                   im_ls = cat(2, im_ls, im_ls(:, end-diffsize(2)+1:end, :));
               case 3
                   im_ls = cat(3, im_ls, im_ls(:, :, end-diffsize(3)+1:end));
           end
        end
    end
    
    % Difference between the original low resolution image the the low resolution image from
    % recovered high resolution image
    im_diff = im_l - im_ls;
    
    % im_diff = imresize(im_diff, [row_h, col_h], 'bicubic'); % original code
    im_diff = single(affine(im_diff, diag([upscale,1]), [], 0)); % turnoff verbose
    
    
    % Make sure the dimension is consistant
    size_im_diff = size(im_diff);
    if min(size_im_h./size_im_diff) < 1
        im_diff = im_diff(1:size_im_h(1),1:size_im_h(2),1:size_im_h(3));
        size_im_diff = size(im_diff);
    end
    if max(size_im_h./size_im_diff) > 1
        diffsize = size_im_h - size_im_diff;
        for ii = 1:length(diffsize)
           if diffsize(ii) == 0
               continue;
           end
           switch ii
               case 1
                   im_diff = cat(1, im_diff, im_diff(end-diffsize(1)+1:end, :, :));
               case 2
                   im_diff = cat(2, im_diff, im_diff(:, end-diffsize(2)+1:end, :));
               case 3
                   im_diff = cat(3, im_diff, im_diff(:, :, end-diffsize(3)+1:end));
           end
        end
    end
    
    % Add the difference back
    % im_h = im_h + conv2(im_diff, p, 'same'); % original code
    im_h = im_h + imgaussfilt3(im_diff, 1, 'FilterSize', 5);  % 2017/01/03: filter by 5x5x5 gaussian filter with std = 1
end
end