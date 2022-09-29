from split_preprocess_augment import *
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
import pickle
from pydicom import dcmread
from us_augment import get_drs_rectangles, us_ratio_augment, select_rectangles
import json

'''
INITIALIZATION: PARAMS AND CHECKING
'''
# Set parameters for creating dataset, for empty values use np.nan
params = dict({'dataset_name': 'us_rect_checked_soft',  # Define name of dataset
               'train_size': 0.8,  # Training set size, [0, 1]
               'val_size': 0.2,  # Validation set size, [0, 1]
               'test_size': 0.0,  # Test set size, [0, 1]. Set to 0 if K-fold = True
               'us_augment': 'projective',  # US augmentation method
               'us_x_dev': 9,  # X-deviation for projective augmentation
               'us_y_dev': 9,  # Y-deviation for projective augmentation
               'ratio_cap': 10,  # Limit of how much times samples can be augmented
               'random_state': 25,  # Random state for splitting for recreating datasets
               'augment_train': True,  # Switch for training set augmentation
               'augment_val': False,  # Switch for validation set augmentation
               'augment_test': False,  # Switch for test set augmentation
               'normalization_us': 'magnitude',  # Normalization method for US
               'exclusion': 'hard',  # Exclusion criteria
               'k-fold': True,  # Prepare dataset for K-fold training
               'notes': 'Checked dataset with relative labels. Soft exclusion criteria so doubtful samples are '
                        'included'})  # Notes for logging

# If K-fold is selected, set test_size to 0. Validation set becomes test set
if (params['k-fold']) & (params['test_size'] != 0):
    params['test_size'] = 0.0
'''
LOAD METADATA
'''
# Load labels and metadata
label_meta = pd.read_csv('Data/table_tot_crop_checked.csv')
label_meta['RelLabel'] = np.full(shape=label_meta.shape[0], fill_value=0) # Create column for relative labels

'''
LOAD AND PRE-PROCESS ULTRASOUND DATA
'''

# Load original US images
us_imgs = loadmat('Data/US_imgs_tot.mat')
us_imgs = us_imgs['US_imgs_tot']

if params['normalization_us'] == 'magnitude':
    us_imgs = norm_us(us_imgs)

# Plot US image corresponding to image
# Get pixel spacing in mm/px, same for both x and y
pixel_spacing = dcmread('Data/example_DICOM_US')[0x18, 0x6011][0]['PhysicalDeltaX'].value * 10  # in mm/px
banana_width = np.int32(np.round(2.5 / pixel_spacing))  # Banana width is about 5 mm, so 2.5 per side
measure_depth = np.int32(np.round(10 / pixel_spacing))  # Maximum measure depth is about 10 mm

# Get list of rectangles of DRS location with height=measure_depth and width=2*banana_width combined
# with drs location and depth of gel_layer/contact layer
rect_pts = get_drs_rectangles(label_meta, label_meta['GelY'], banana_width, measure_depth, us_width=us_imgs.shape[-1])

# Define rectangle shape as destination for the projective transform augmentation
dst_pts = np.float32([[0, measure_depth],
                      [0, 0],
                      [banana_width * 2, 0],
                      [banana_width * 2, measure_depth]])

''''
CLEAN UP METADATA: Switch spectra of switched DRS locations 
'''
# Shuffle DRS and US data according to data revision
us_imgs = us_imgs[label_meta['US_index'] - 1]

# Switch around PA labels according to data revision
all_labels = np.copy(label_meta[['PA_LabelLayer1', 'PA_LabelLayer2', 'PA_LabelLayer3']].to_numpy())
all_labels = all_labels[label_meta['PA_index'] - 1]
label_meta[['PA_LabelLayer1', 'PA_LabelLayer2', 'PA_LabelLayer3']] = all_labels

'''
EXCLUDE SAMPLES
Is based on data_sort column in metadata where:
1 = good
2 = doubtful
3 = Exclude
4 = Exclude US but include DRS
5 = Exclude DRS but include US rectangle
'''

if params['exclusion'] == 'hard':  # Hard exclusion leaves no doubful examples
    exclude_idx_us = (label_meta['data_sort'] == 3) | (label_meta['data_sort'] == 4) | (label_meta['data_sort'] == 2)
    us_imgs = np.delete(us_imgs, label_meta.index[exclude_idx_us], axis=0)
    rect_pts = np.delete(rect_pts, label_meta.index[exclude_idx_us], axis=0)
    label_meta = label_meta.drop(label_meta.index[exclude_idx_us])

elif params['exclusion'] == 'soft':  # Soft exclusion leaves doubtful examples
    exclude_idx_us = (label_meta['data_sort'] == 3) | (label_meta['data_sort'] == 4)
    us_imgs = np.delete(us_imgs, label_meta.index[exclude_idx_us], axis=0)
    rect_pts = np.delete(rect_pts, label_meta.index[exclude_idx_us], axis=0)
    label_meta = label_meta.drop(label_meta.index[exclude_idx_us])
else:
    print('No exclusion is applied')

# Get patient ids
pat_ids = label_meta['PatientID'].to_numpy()

# Select pt prefix from PatientIDs
for count, item in enumerate(pat_ids):
    pat_ids[count] = item.split('_')[0]

'''
SPLIT DATA
'''

# Get stratification relative to first two layers and set relative labels in metadata
labels_stratify, pat_stratify, label_meta = get_relative_stratify_labels(label_meta, pat_ids)

# Select all relative labels labels
labels_all = label_meta['RelLabel'].to_numpy()

# Get all labels and remove NaN-values from array
labels_all = labels_all.copy()

# Remove singular full tumor sample to force it in training set
labels_stratify = np.delete(labels_stratify, np.where(pat_stratify == 'Ex39'))
pat_stratify = np.delete(pat_stratify, np.where(pat_stratify == 'Ex39'))

# Split Patients based on combinations present in the first two layers of the data
train_ids, val_ids, test_ids = split_data(pat_ids=pat_stratify,
                                          labels=labels_stratify,
                                          train_size=params['train_size'],
                                          val_size=params['val_size'],
                                          test_size=params['test_size'],
                                          random_state=params['random_state'])

# Force single 'all tumor' occurrence in train set
train_ids = np.append(train_ids, 'Ex39')

'''
SELECT DATA BASED ON SPLITTING IDs
'''

# Get indices of selected train, test and val ids
train_idx = ids2indices(pat_ids, train_ids)
val_idx = ids2indices(pat_ids, val_ids)
test_idx = ids2indices(pat_ids, test_ids)

# Select labels
train_labels, val_labels, test_labels = labels_all[train_idx], labels_all[val_idx], labels_all[test_idx]

# Select meta data, fiber data, US images and labels with partition indices
train_meta, val_meta, test_meta = label_meta.iloc[train_idx], label_meta.iloc[val_idx], label_meta.iloc[test_idx]
train_us, val_us, test_us = us_imgs[train_idx], us_imgs[val_idx], us_imgs[test_idx]
train_rects, val_rects, test_rects = rect_pts[train_idx], rect_pts[val_idx], rect_pts[test_idx]

# Select IDs in same order as data as IDs and indices are shuffled during splitting
train_ids_sort, val_ids_sort, test_ids_sort = pat_ids[train_idx], pat_ids[val_idx], pat_ids[test_idx]

'''
AUGMENTATION OF US
'''
train_sample_ids, val_sample_ids, test_sample_ids = \
    train_meta['Sample_id'].to_numpy(), val_meta['Sample_id'].to_numpy(), test_meta['Sample_id'].to_numpy()

# Augment training set based on ratios of present layer combinations in the first two layers
if params['augment_train']:
    labels_train_ratios, labels_train_idx = get_label_ratios(train_labels, ratio_cap=params['ratio_cap'])
    train_us, train_labels, train_ids_augment, train_sample_ids = us_ratio_augment(labels_train_ratios,
                                                                                   labels_train_idx, train_rects,
                                                                                   dst_pts, train_us,
                                                                                   params['us_x_dev'],
                                                                                   params['us_y_dev'], train_labels,
                                                                                   train_ids_sort, train_sample_ids,
                                                                                   show=False)


else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    train_us = select_rectangles(train_rects, train_us, dst_pts)
    train_ids_augment = train_ids_sort

if params['augment_val']:
    labels_val_ratios, labels_val_idx = get_label_ratios(val_labels, ratio_cap=params['ratio_cap'])
    val_us, val_labels, val_ids_augment, val_sample_ids = us_ratio_augment(labels_val_ratios, labels_val_idx, val_rects,
                                                                           dst_pts, val_us,
                                                                           params['us_x_dev'], params['us_y_dev'],
                                                                           val_labels, val_ids_sort, val_sample_ids,
                                                                           show=False)
    print('Augmented Validation set')
else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    val_us = select_rectangles(val_rects, val_us, dst_pts)
    val_ids_augment = val_ids_sort

if params['augment_test']:
    labels_test_ratios, labels_test_idx = get_label_ratios(test_labels, ratio_cap=params['ratio_cap'])
    test_us, test_labels, test_ids_augment, test_sample_ids = us_ratio_augment(labels_test_ratios, labels_test_idx,
                                                                               test_rects, dst_pts, test_us,
                                                                               params['us_x_dev'], params['us_y_dev'],
                                                                               test_labels, test_ids_sort,
                                                                               test_sample_ids, show=False)
    print('Augmented Test set')
else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    test_us = select_rectangles(test_rects, test_us, dst_pts)
    test_ids_augment = test_ids_sort

'''
SAVING DATASET
'''

data_dict = dict({'train_us': train_us,
                  'val_us': val_us,
                  'test_us': test_us,
                  'train_meta': train_meta,
                  'val_meta': val_meta,
                  'test_meta': test_meta,
                  'train_labels': train_labels,
                  'val_labels': val_labels,
                  'test_labels': test_labels,
                  'params': params,
                  'train_ids_augment': train_ids_augment,
                  'val_ids_augment': val_ids_augment,
                  'test_ids_augment': test_ids_augment,
                  'labels_stratify': labels_stratify,
                  'pat_stratify': pat_stratify,
                  'train_ids': train_ids,
                  'train_sample_ids': train_sample_ids,
                  'val_sample_ids': val_sample_ids,
                  'test_sample_ids': test_sample_ids
                  })

# Check to prevent overwriting of previous dataset
save_name = os.path.join('Data processed/' + params['dataset_name'] + '.pkl')
if not os.path.exists(save_name):
    with open(save_name, 'wb') as file:
        pickle.dump(data_dict, file)
    # Save accompanying json file for dataset registration
    with open('Data processed/' + params['dataset_name'] + '_params' + '.json', 'w') as file:
        json.dump(params, file, indent=4)
else:
    raise FileExistsError('Dataset with this name already created, choose another name')
