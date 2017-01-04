% Demo AlexNet
cnnURL = 'http://www.vlfeat.org/matconvnet/models/beta22/imagenet-caffe-alex.mat';

% Store CNN model in a temporary folder
cnnMatFile = fullfile(tempdir, 'imagenet-caffe-alex.mat');
if ~exist(cnnMatFile, 'file') % download only once
    disp('Downloading pre-trained CNN model...');
    websave(cnnMatFile, cnnURL);
end

