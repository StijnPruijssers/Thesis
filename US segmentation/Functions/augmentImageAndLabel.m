%% Methods
function data = augmentImageAndLabel(data, x_reflect, gamma_range)
% Augment images and pixel label images using random reflection and
% translation.

for i = 1:size(data,1)
    
    tform = randomAffine2d('XReflection', x_reflect);
    
    % Center the view at the center of image in the output space while
    % allowing translation to move the output image out of view.
    rout = affineOutputView(size(data{i,1}), tform, 'BoundsStyle', 'centerOutput');
    
    % Warp the image and pixel labels using the same transform.
    data{i,1} = imwarp(data{i,1}, tform, 'OutputView', rout);
    data{i,2} = imwarp(data{i,2}, tform, 'OutputView', rout);

    rand_gamma = diff(gamma_range) * rand() + gamma_range(1);       % Select random number for gamma correction
    data{i,1} = gammaCorrection(data{i,1}, rand_gamma);

    
end
end
