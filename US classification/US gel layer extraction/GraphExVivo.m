%% Graph Theory - Ex vivo

function [Layers,Top,Grad, GelY] = GraphExVivo(img,params)
% BASED ON CASEREL IMPLEMENTATION OF PANGYU TENG
% $Created:  1.0 $ $Date: 2013/09/09 20:00$ $Author: Pangyu Teng $
% $Revision: 1.1 $ $Date: 2013/09/15 21:00$ $Author: Pangyu Teng $

% Required inputs:
%  - params.DRSposition (1, 2 or 3)
%  - params.PixelSize (in mm)
%  - params.Region (depth of interest in mm)


if nargin < 1
    disp('requires 1 input');
    return;
end

% Define parameters
params.isResize = [false 0.5];  % resize the image if 1st value set to 'true',
                                % with the second value to be the scale.
params.filter0Params = [params.Gauss params.Gauss 1]; % parameter for smoothing the images.
params.isPlot = true;     
params.txtOffset = -7;

% Create figure before colormap
if params.isShow
    f = figure('visible', 'on');
else
    f = figure('visible', 'off');
end

params.colorarr=colormap('jet'); 

% Clear up matlab's mind
clear Layers
ImgOriginal = img;

% Get image size
szImg = size(img);

% Resize image.
if params.isResize(1)
    img = imresize(img,params.isResize(2),'bilinear');
end

% Smooth image with specified kernels for denosing
img = imfilter(img,fspecial('gaussian',params.filter0Params(1:2),params.filter0Params(3)),'replicate');        

% Create adjacency matrices and its elements base on the image.
[params.adjMatrixW, params.adjMatrixMW, params.adjMA, params.adjMB, params.adjMW, params.adjMmW, gradImgY, imgNew] = getAdjacencyMatrix(img);

% Layers to find
LayerOrder = {'ContactSurface','secondLayer'};

% Segment layers
Layers = [];
for layerInd = 1:numel(LayerOrder)   
    if layerInd == 1
        [Layers, ~] = getLayers(LayerOrder{layerInd},imgNew,params,Layers);
    elseif layerInd > 1
        for i = 1:1
            [Layers, ~] = getLayers(LayerOrder{layerInd},imgNew,params,Layers,i);
        end
    end
end

% Delete elements of the adjacency matrices prior function exit to save memory
toBeDeleted = {'adjMatrixWSmo' 'adjMatrixMWSmo' 'adjMWSmo' 'adjMmWSmo'  'adjMW' 'adjMmW' 'adjMatrixW' 'adjMatrixMW' 'adjMA' 'adjMB'};
for delInd = 1:numel(toBeDeleted)
    params.(toBeDeleted{delInd}) = [];
end

% Plot selected paths
if params.isPlot
    imagesc(ImgOriginal);
    axis image; colormap('gray'); hold on; drawnow;

    layersToPlot = {'ContactSurface','secondLayer'};%
    for k = 1:numel(layersToPlot)

        matchedLayers = strcmpi(layersToPlot{k},{Layers(:).name});
        layerToPlotInd = find(matchedLayers == 1);
        for i = 1:numel(layerToPlotInd)
            if ~isempty(Layers(layerToPlotInd(i)).pathX)
                if k == 1
                    colora = params.colorarr(i,:);
                elseif k == 2
                    colora = params.colorarr(i*20,:);
                end
                plot(Layers(layerToPlotInd(i)).pathY,Layers(layerToPlotInd(i)).pathX-1,'--','color',colora,'linewidth',1.5);           
                
                top_ind = find(Layers(2).pathY == round(size(img, 2)/2));
                GelY = Layers(2).pathX(top_ind);
                plot(round(size(img, 2)/2), GelY, '*r')
                
                drawnow;
            end % of if ~isempty            
        end
    end % of k
    hold off;
    if params.isSave
        saveas(f, sprintf('Images_edge/edge_entry_%i.png', params.iteration+394))
        close(f)
    end
end % of isPlot 

% Calculate Top Layer Thickness and Gradient at DRS position
nLayers = numel(Layers)-1;
Top  = NaN(nLayers,1);
Grad = NaN(nLayers,1);

for k = 2:nLayers+1
    if ~isempty(Layers(k).path)
        switch params.DRSposition
            case 'DRS1'
                pointer1  = find(Layers(1).pathY == round(szImg(2)*0.15));
                pointer2  = find(Layers(k).pathY == round(szImg(2)*0.15));
                Grad(k-1) = max(gradImgY(Layers(k).pathX(pointer2(1))-2:Layers(k).pathX(pointer2(1))+2,round(szImg(2)*0.15)));
                Top(k-1)  = (Layers(k).pathX(pointer2(1))-Layers(1).pathX(pointer1(1)))*params.PixelSize;
            case 'DRS2'
                pointer1  = find(Layers(1).pathY == round(szImg(2)*0.5));
                pointer2  = find(Layers(k).pathY == round(szImg(2)*0.5));
                Grad(k-1) = max(gradImgY(Layers(k).pathX(pointer2(1))-2:Layers(k).pathX(pointer2(1))+2,round(szImg(2)*0.5)));
                Top(k-1)  = (Layers(k).pathX(pointer2(1))-Layers(1).pathX(pointer1(1)))*params.PixelSize;
            case 'DRS3'
                pointer1  = find(Layers(1).pathY == round(szImg(2)*0.85));
                pointer2  = find(Layers(k).pathY == round(szImg(2)*0.85));
                Grad(k-1) = max(gradImgY(Layers(k).pathX(pointer2(1))-2:Layers(k).pathX(pointer2(1))+2,round(szImg(2)*0.85)));
                Top(k-1)  = (Layers(k).pathX(pointer2(1))-Layers(1).pathX(pointer1(1)))*params.PixelSize;                
        end
    end
end
end

%%
function [adjMatrixW, adjMatrixMW, adjMAsub, adjMBsub, adjMW, adjMmW, gradImgY, img] = getAdjacencyMatrix(inputImg)
    %
    % Ouputs the adjacency matrices, and the weights and locations for building these matrices based on eq 1
    % of following article with the input image 'imgOld'
    %
    % Chui SJ et al, Automatic segmentation of seven retinal layers in SDOCT
    % images congruent with expert manual segmentation, Optics Express, 2010;18(18);19413-19428
    % Section 3.1 to 3.3
    % link(pubmed): http://goo.gl/Z8zsY
    % link(pdf from Duke.edu): http://goo.gl/i3cJ0
    %
    % Usage
    % Input: 'inputImg' - an image.
    % Outputs: 
    %   %Sparse Matrices
    %   'adjMatrixW'    -  dark-to-light adjacency matrix
    %   'adjMatrixMW'   - light-to-dark adjacency matrix
    %   %Non zero weights in the sparse matrices
    %   'adjMBsub'      - locations of the weights
    %   'adjMAsub'      - locations of the weights
    %   'adjMmW'        - light-to-dark weights
    %   'adjMW'         - dark-to-light weights
    %   'img'           - updated image with 1 column of 0s on the each verticle side of the image
    %
    % $Revision: 1.0 $ $Date: 2013/04/29 09:00$ $Author: Pangyu Teng $
    % $Revision: 1.1 $ $Date: 2013/09/15 21:00$ $Author: Pangyu Teng $

    % pad image with vertical column on both sides
    sizeImg = size(inputImg);
    img = zeros([sizeImg(1) sizeImg(2)+2]);

    img(:,2:1+sizeImg(2)) = inputImg;

    % update size of image
    sizeImg = size(img);

    % get gradient images - Original: pixelwise 
    [~,gradImgY] = gradient(img,1,1);
    
    gradImgY = -1*gradImgY;
    
    % normalize gradients
    gradImgY = (gradImgY-min(gradImgY(:)))/(max(gradImgY(:))-min(gradImgY(:)));
    
    % get the "invert" of the gradient image.
    gradImgMinusY = gradImgY*-1+1; 
    
    % generate adjacency matrix, see equation 1 in the refered article.

    % minimum weight
    minWeight = 1E-5;

    neighborIterX = [1 1  1 0  0 -1 -1 -1];
    neighborIterY = [1 0 -1 1 -1  1  0 -1];

    % get location A (in the image as indices) for each weight.
    adjMAsub = 1:sizeImg(1)*sizeImg(2);

    % convert adjMA to subscripts
    [adjMAx,adjMAy] = ind2sub(sizeImg,adjMAsub);

    adjMAsub   = adjMAsub';
    szadjMAsub = size(adjMAsub);

    % prepare to obtain the 8-connected neighbors of adjMAsub
    % repmat to [1,8]
    neighborIterX = repmat(neighborIterX, [szadjMAsub(1),1]);
    neighborIterY = repmat(neighborIterY, [szadjMAsub(1),1]);

    % repmat to [8,1]
    adjMAsub = repmat(adjMAsub,[1 8]);
    adjMAx   = repmat(adjMAx,  [1 8]);
    adjMAy   = repmat(adjMAy,  [1 8]);

    % get 8-connected neighbors of adjMAsub
    % adjMBx,adjMBy and adjMBsub
    adjMBx = adjMAx+neighborIterX(:)';
    adjMBy = adjMAy+neighborIterY(:)';

    % make sure all locations are within the image.
    keepInd = adjMBx > 0 & adjMBx <= sizeImg(1) & ...
        adjMBy > 0 & adjMBy <= sizeImg(2);

    % adjMAx = adjMAx(keepInd);
    % adjMAy = adjMAy(keepInd);
    adjMAsub = adjMAsub(keepInd);
    adjMBx = adjMBx(keepInd);
    adjMBy = adjMBy(keepInd); 

    adjMBsub = sub2ind(sizeImg,adjMBx(:),adjMBy(:))';

    % calculate weight
    adjMW  = 2 - gradImgY(adjMAsub(:)) - gradImgY(adjMBsub(:)) + minWeight;
    adjMmW = 2 - gradImgMinusY(adjMAsub(:)) - gradImgMinusY(adjMBsub(:)) + minWeight;

    % pad minWeight on the side
    imgTmp = nan(size(gradImgY));
    imgTmp(:,1) = 1;
    imgTmp(:,end) = 1;
    imageSideInd = ismember(adjMBsub,find(imgTmp(:)==1));
    adjMW(imageSideInd) = minWeight;
    adjMmW(imageSideInd) = minWeight;

    % build sparse matrices
    adjMatrixW = sparse(adjMAsub(:),adjMBsub(:),adjMW(:),numel(img(:)),numel(img(:)));
    % build sparse matrices with inverted gradient.
    adjMatrixMW = sparse(adjMAsub(:),adjMBsub(:),adjMmW(:),numel(img(:)),numel(img(:)));
end

%%
function [rPaths, img] = getLayers(layerName,img,params,rPaths,i)

%   $Created: 1.0 $ $Date: 2013/09/09 20:00$ $Author: Pangyu Teng $
%   $Revision: 1.1 $ $Date: 2013/09/15 21:00$ $Author: Pangyu Teng $
if nargin < 3
    disp('3 inputs required, getLayers.m');
    return;   
end

adjMA  = params.adjMA;
adjMB  = params.adjMB;
adjMW  = params.adjMW;
adjMmW = params.adjMmW;

switch layerName
    case {'ContactSurface'}       
        BD = 1; % Bright to dark
        it = 0;
    case {'secondLayer'}
        BD = 0; % Bright to dark
        it = i;
end

% initialize region of interest
ImgSize = size(img);
roiImg = zeros(ImgSize);

% select region of interest based on layers priorly segmented.
for k = 2:ImgSize(2)-1
    
    switch layerName
        
        case {'ContactSurface'}
              startInd = 1;
              endInd = 15;  % Contact surface assumed to be within first 15 pixels
            
        case {'secondLayer'}

            indPathX = find(rPaths(strcmp('ContactSurface',{rPaths.name})).pathY==k);
            
            % define a region from contact surface until params.Region mm below
            startInd0 = rPaths(strcmp('ContactSurface',{rPaths.name})).pathX(indPathX(1));
            startInd = startInd0+10;
            endInd = startInd0+ceil(params.Region/params.PixelSize);
            
            % mask regions in which a path was already selected
            if it > 1
                if isempty(rPaths(end).path)
                    return;
                else
                layerIDX = strcmp({'secondLayer'},{rPaths.name});
                layerIDX = find(layerIDX == 1);
                excludeInd = NaN(numel(layerIDX),1);
                for i = 1:numel(layerIDX)
                    indPathX = find(rPaths(layerIDX(i)).pathY==k);
                    excludeInd(i) = rPaths(layerIDX(i)).pathX(indPathX(1));
                end
                end
            end
    end
    
    %error checking    
    if startInd > endInd
        startInd = endInd - 1;
    end            
    
    if startInd < 1
        startInd = 1;
    end
    
    if endInd > ImgSize(1)
        endInd = ImgSize(1);
    end
    
    if it > 1 && sum(excludeInd+10 > endInd)>0        
        excludeInd(excludeInd+10 > endInd) = endInd-10;
    end
                    
    % set region of interest at column k from startInd to endInd
    roiImg(startInd:endInd,k) = 1;
    if it > 1
        for i = 1:numel(excludeInd)
            roiImg(excludeInd(i)-10:excludeInd(i)+10,k) = 0;
        end
    end
end

% ensure the 1st and last column is part of the region of interest.
roiImg(:,1)=1;
roiImg(:,end)=1;            

% include only region of interst in the adjacency matrix
includeA = ismember(adjMA, find(roiImg(:) == 1));
includeB = ismember(adjMB, find(roiImg(:) == 1));
keepInd  = includeA & includeB;

if BD == 1
    adjMatrixW  = sparse(adjMA(keepInd),adjMB(keepInd),adjMW(keepInd),numel(img(:)),numel(img(:)));
    [ ~, path ] = graphshortestpath(adjMatrixW, 1, numel(img(:)));
elseif BD == 0
    adjMatrixMW = sparse(adjMA(keepInd),adjMB(keepInd),adjMmW(keepInd),numel(img(:)),numel(img(:)));
    [ ~, path ] = graphshortestpath(adjMatrixMW, 1, numel(img(:)));
end

%convert path indices to subscript
[pathX, pathY] = ind2sub(ImgSize,path);
    
LayerIDX = numel(rPaths)+1;
rPaths(LayerIDX).name = layerName;

% save data.
rPaths(LayerIDX).path  = path;
rPaths(LayerIDX).pathX = pathX;
rPaths(LayerIDX).pathY = pathY;
rPaths(LayerIDX).pathXmean = mean(rPaths(LayerIDX).pathX(gradient(rPaths(LayerIDX).pathY)~=0));

end