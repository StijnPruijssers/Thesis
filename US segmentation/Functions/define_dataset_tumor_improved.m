function [train_ds, val_ds, test_ds] = define_dataset_tumor_improved(aug, gamma_range, x_reflect)
    
    %% Create datastores
    parent_dir = 'C:\Users\s.pruijssers\Transfer learning';
    img_dir = fullfile(parent_dir,'Images_final');
    label_dir = fullfile(parent_dir,'Labels_final');


    class_names = ["normal" "tumor"];
    pixel_ids = [0, 1];
    
    train_imgds = imageDatastore(fullfile(img_dir, 'Train'), 'FileExtensions','.png');
    val_imgds = imageDatastore(fullfile(img_dir, 'Val'), 'FileExtensions','.png');
    test_imgds = imageDatastore(fullfile(img_dir, 'Test'), 'FileExtensions','.png');
    
    train_pxds = pixelLabelDatastore(fullfile(label_dir, 'Train'), class_names, pixel_ids, 'FileExtensions','.png');
    val_pxds = pixelLabelDatastore(fullfile(label_dir, 'Val'), class_names, pixel_ids, 'FileExtensions','.png');
    test_pxds = pixelLabelDatastore(fullfile(label_dir, 'Test'), class_names, pixel_ids, 'FileExtensions','.png');
    
    % Combine datastores
    train_ds = combine(train_imgds, train_pxds);
    val_ds = combine(val_imgds, val_pxds);
    test_ds = combine(test_imgds, test_pxds);
    
    %% Augmentation
    if aug
        gamma_range = [0.75, 1.25];
        x_reflect = true;
        train_ds = transform(train_ds, @(data)augmentImageAndLabel(data,x_reflect,gamma_range));
    end

end


