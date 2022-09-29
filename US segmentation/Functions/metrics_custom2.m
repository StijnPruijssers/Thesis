function [metrics, ME_top, ME_per_sample, std_ME] = metrics_custom2(preds, labels)

    confmats = cell(size(labels, 3), 1);
    for i=1:size(labels, 3)
        g1 = labels(:, :, i);
        g2 = preds(:,:, i);
        confmats{i} = confusionmat(g1(:),g2(:));
    end
                
    % Get Semantic segmentation metrics
    metrics =  evaluateSemanticSegmentation(confmats, ["normal" "tumor"]);
    

    % Get RMSE of top pixel of prediction
    peak_pred = zeros([size(labels, 3), 1]);
    peak_gt = zeros([size(labels, 3), 1]);

    for i=1:size(labels, 3)
        pred_bw = preds(:,:,i);
        gt_bw = labels(:,:,i);

        pred_bw = bwareafilt(pred_bw, 1);    % Select largest connected component
        [row_pred, ~] = ind2sub(size(pred_bw), find(pred_bw==1));
        if isempty(min(row_pred)) % If prediction is empty set it to max distance 
            peak_pred(i) = 0;
            peak_gt(i) = 128;

        else
            peak_pred(i) = min(row_pred);

            [row_gt, ~] = ind2sub(size(gt_bw), find(gt_bw==1));

            
            peak_gt(i) = min(row_gt);
        end
    end
    ME_per_sample = [peak_gt, peak_pred];
    ME_top = mean(abs(peak_gt - peak_pred));
    std_ME = std(abs(peak_gt - peak_pred));

end