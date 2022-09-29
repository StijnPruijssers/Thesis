% Script for extracting the Y-value of the transition from gel-layer to
% tissue. Uses an adjusted version of the algorithm by Chui SJ et
% al, Automatic segmentation of seven retinal layers in SDOCT images
% congruent with expert manual segmentation, Optics Express,
% 2010;18(18);19413-19428
% link(pubmed): http://goo.gl/Z8zsY
%
% BASED ON CASEREL IMPLEMENTATION OF PANGYU TENG $Created:  1.0 $ $Date:
% 2013/09/09 20:00$ $Author: Pangyu Teng $ $Revision: 1.1 $ $Date:
% 2013/09/15 21:00$ $Author: Pangyu Teng $


%% Load data 
load('US_imgs_tot.mat')

%% Show and measure phase 1 images manually
x = zeros(1, size(US_imgs_tot, 1));
y = zeros(1, size(US_imgs_tot, 1));

for i=1:size(US_imgs_tot, 1)
    fig = figure;
    imshow(squeeze(US_imgs_tot(i,:,:)))
    [x(i), y(i)] = getpts
    close(fig)
end


%% Phase 2 gel extraction
% Select all images with a Gel layer present (Phase 2)
US_imgs_tot = US_imgs_tot(395:end, :, :);
test_img = squeeze(US_imgs_tot(1, :, :)); % Remove redundant dimension

% Create parameter struct graph-cut algorithm
params = struct();
params.DRSposition = 2; % 1 = left, 2 = middle, 3 = right
params.PixelSize = 0.066; % pixel spacing 
params.Region = 20; % region to search for Gel Layer after first 15 pixels in mm 

%% Graph-cut single image as example
% Dont Save, just show
params.isShow = true;
params.isSave = false;
params.Gauss = 5; % Set amount of Gaussian filtering before gradient image

% Use graph-cut method by Chiu et al. for layer segmentation
[Layers, Top, Grad] = GraphExVivo(squeeze(US_imgs_tot(1, :, :)),params);
%% Graph-cut all
% Dont show, just save or combination
params.isShow = false;
params.isSave = true;

GelY_vals = zeros(1, size(US_imgs_tot, 1)); % Pre-allocate

% For every image
for i=1:size(US_imgs_tot, 1)
    params.iteration = i; % extra parameter for saving
    disp(i) 
    [Layers, Top, Grad, GelY] = GraphExVivo(squeeze(US_imgs_tot(i, :, :)),params);
    GelY_vals(i) = GelY;
end

% Save found GelY values
save('GelY_vals.mat', 'GelY_vals')

%% Graph-cut all unique images
US_imgs_unique = load('TumorOrNot2.mat');
US_imgs_unique = US_imgs_unique.SingleUSimages;

% Dont show, just save or combination
params.isShow = true;
params.isSave = false;

GelY_vals = zeros(1, size(US_imgs_unique, 1)); % Pre-allocate

% For every image
for i=1:size(US_imgs_unique, 1)
    params.iteration = i; % extra parameter for saving
    disp(i) 
    [Layers, Top, Grad, GelY] = GraphExVivo(squeeze(US_imgs_tot(i, :, :)),params);
    GelY_vals(i) = GelY;
end





