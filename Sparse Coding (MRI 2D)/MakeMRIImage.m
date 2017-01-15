MRI_template = load_untouch_nii('D:\MRI Data\Cardiac Data\3D Model\time_1\3D_Model_truncate.nii');
id = randperm(95);
for i = 1:85
    s = single(squeeze(MRI_template.img(id(i),:,:)));
    s = s - min(s(:));
    s = uint16(s / max(s(:)) * 32767);
    filename = ['train', num2str(i), '.png'];
    imwrite(s,filename);
end

for i = 86:95
    s = single(squeeze(MRI_template.img(id(i),:,:)));
    s = s - min(s(:));
    s = uint16(s / max(s(:)) * 32767);
    filename = ['valid', num2str(i-85), '.png'];
    imwrite(s,filename);
end