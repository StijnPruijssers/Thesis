from split_preprocess_augment import *
from utils import drs_calc_loss_weights
import os
import numpy as np
from scipy.io import loadmat
import pandas as pd
import pickle
from split_preprocess_augment import snv, norm_800, magnitude_norm
import json

'''
INITIALIZATION: PARAMS AND CHECKING
'''

# Set parameters for creating dataset, for empty values use np.nan
params = dict({'dataset_name': 'drs_checked_hard',      # Define name of dataset
               'drs_down_rate': np.nan,                 # To which number of 'wavelenghts' spectra are downsampled
               'train_size': 0.8,                       # Training set size, [0, 1]
               'val_size': 0.2,                         # Validation set size, [0, 1]
               'test_size': 0.0,                        # Test set size, [0, 1]. Set to 0 if K-fold = True
               'drs_stretch_delta': 0.1,                # Amount of stretching in DRS augmentation as percentage of STD
               'drs_offset_delta': 0.1,                 # Amount of offset in DRS augmentation as percentage of STD
               'drs_slope_delta': 0.05,                 # Amount of slope adjustment in DRS augmentation as
                                                        #    percentage of current slope
               'ratio_cap': 10,                         # Limit of how much times samples can be augmented
               'random_state': 25,                      # Random state for splitting for recreating datasets
               'augment_train': True,                   # Switch for training set augmentation
               'augment_val': False,                    # Switch for validation set augmentation
               'augment_test': False,                   # Switch for test set augmentation
               'normalization': 'raw',                  # Normalization method for DRS
               'exclusion': 'hard',                    # Exclusion criteria
               'k-fold': True,                          # Prepare dataset for K-fold training
               'notes': 'Checked dataset with relative labels. Hard exclusion criteria so no doubtful samples are '
                        'included'})                    # Notes for logging

# If K-fold is selected, set test_size to 0. Validation set becomes test set
if (params['k-fold']) & (params['test_size'] != 0):
    params['test_size'] = 0.0

'''
LOAD METADATA
'''
# Load labels and metadata
label_meta = pd.read_csv('Data/table_tot_crop_checked.csv')
label_meta['RelLabel'] = np.full(shape=label_meta.shape[0], fill_value=0)   # Create column for relative labels

'''
LOAD AND PRE-PROCESS FIBER DATA
'''

# Load original fiber data
fib_data = loadmat('Data/fib_tot.mat')
fib_data = fib_data['fib_tot']

# Interpolate sample if indicated
if not np.isnan(params['drs_down_rate']):
    fib_data = interpolate_array1d(fib_data, params['drs_down_rate'])

# Normalize fiber data if indicated
if params['normalization'] == 'magnitude':
    fib_data = magnitude_norm(fib_data)
elif params['normalization'] == 'snv':
    fib_data = snv(fib_data)
elif params['normalization'] == '800nm':
    fib_data = norm_800(fib_data)


''''
CLEAN UP METADATA: Switch spectra of switched DRS locations 
'''
# Shuffle DRS and US data according to data revision
fib_data = fib_data[label_meta['DRS_index'] - 1]

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
    exclude_idx_drs = (label_meta['data_sort'] == 3) | (label_meta['data_sort'] == 5) | (label_meta['data_sort'] == 2)
    fib_data = np.delete(fib_data, label_meta.index[exclude_idx_drs], axis=0)
    label_meta = label_meta.drop(label_meta.index[exclude_idx_drs])

elif params['exclusion'] == 'soft':  # Soft exclusion leaves doubtful examples
    exclude_idx_drs = (label_meta['data_sort'] == 3) | (label_meta['data_sort'] == 5)
    fib_data = np.delete(fib_data, label_meta.index[exclude_idx_drs], axis=0)
    label_meta = label_meta.drop(label_meta.index[exclude_idx_drs])
else:
    print('No exclusion is applied')


'''
CALCULATE LOSS WEIGHTS BASED ON INTENSITIES
'''
loss_weights = drs_calc_loss_weights(fib_data, label_meta)

'''
SPLIT DATA
'''

# Get patient ids
pat_ids = label_meta['PatientID'].to_numpy()

# Select pt prefix from PatientIDs
for count, item in enumerate(pat_ids):
    pat_ids[count] = item.split('_')[0]

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
train_fib, val_fib, test_fib = fib_data[train_idx], fib_data[val_idx], fib_data[test_idx]

# Select IDs in same order as data as IDs and indices are shuffled during splitting
train_ids_sort, val_ids_sort, test_ids_sort = pat_ids[train_idx], pat_ids[val_idx], pat_ids[test_idx]

'''
AUGMENTATION OF DRS
'''
# Create arrays to give each sample an ID. Each augmented sample gets the same ID as the original for comparison
train_sample_ids, val_sample_ids, test_sample_ids = \
    train_meta['Sample_id'].to_numpy(), val_meta['Sample_id'].to_numpy(), test_meta['Sample_id'].to_numpy()

# Augment training set based on ratios of present layer combinations in the first two layers
if params['augment_train']:
    labels_train_ratios, labels_train_idx = get_label_ratios(train_labels, ratio_cap=params['ratio_cap'])
    train_fib, train_labels, train_ids_augment, train_sample_ids = drs_augment(train_labels, train_fib, train_ids_sort,
                                                                               params,
                                                                               labels_train_ratios,
                                                                               labels_train_idx,
                                                                               sample_ids=train_sample_ids)

else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    train_ids_augment = train_ids_sort

if params['augment_val']:
    labels_val_ratios, labels_val_idx = get_label_ratios(val_labels, ratio_cap=params['ratio_cap'])
    val_fib, val_labels, val_ids_augment, val_sample_ids = drs_augment(val_labels, val_fib, val_ids_sort, params,
                                                                       labels_val_ratios,
                                                                       labels_val_idx, sample_ids=val_sample_ids)
    print('Augmented Validation set')
else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    val_ids_augment = val_ids_sort

if params['augment_test']:
    labels_test_ratios, labels_test_idx = get_label_ratios(test_labels, ratio_cap=params['ratio_cap'])
    test_fib, test_labels, test_ids_augment, test_sample_ids = drs_augment(test_labels, test_fib, test_ids_sort, params,
                                                                           labels_test_ratios,
                                                                           labels_test_idx, sample_ids=test_sample_ids)
    print('Augmented Test set')
else:  # If no augmentation is selected, just select US rectangles and assign original ids as augmented ids
    test_ids_augment = test_ids_sort

'''
SAVING DATASET
'''

data_dict = dict({'train_fib': train_fib,
                  'val_fib': val_fib,
                  'test_fib': test_fib,
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
                  'test_sample_ids': test_sample_ids,
                  'loss_weights': loss_weights
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


