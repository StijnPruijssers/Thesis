import os
import pickle
import numpy as np
from scipy.io import loadmat
import pandas as pd
from clodsa.techniques.techniqueFactory import createTechnique
from us_augment import rot_crop_image
import cv2
from split_preprocess_augment import norm_us
from split_preprocess_augment import ids2indices


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# The GPU id to use, usually either 0,1,2 or 3.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['WANDB_DIR'] = 'Z:\Personal folder\Stijn Pruijssers (M3)_UT_2021\Software\Code\CRNN_git3'

run_params = dict({'dataset_name': ['dataset_3'],
                   'hyper_param_name': ['Complete_US_V3'],
                   'label_format': ['binary'],
                   'max_trials': [800],
                   'exec_per_trial': [1],
                   'overwrite': [False],
                   'pc_state': ['server'],
                   'weighting': [True],
                   'n_folds': [4],
                   'random_state': [3],
                   'augment': [False],
                   'normalize': [False]
                   })

# Load original data
tumor_or_not = loadmat('Data/TumorOrNot2.mat')
us_imgs = tumor_or_not['SingleUSimages']
us_imgs = np.transpose(us_imgs, (2, 0, 1))

# Load extracted video frames
with open('Data processed/extracted_frames.pkl', 'rb') as file:
    extract_frames = pickle.load(file)

us_imgs_extract = extract_frames['us_imgs'][:, 0:230, :]

# Crop to bottom half of image
us_imgs_phase1_crop = us_imgs[0:149, 0:230, :]   # Phase 1


gel_mean = np.round(np.squeeze(loadmat('Data/Gel_mean.mat')['Gel_mean'])).astype('uint64') # Load gel layer data
gel_mean = gel_mean - 5
us_imgs_phase2 = np.copy(us_imgs[149::])
us_imgs_phase2_crop = np.zeros(shape=(us_imgs_phase2.shape[0], 230, us_imgs_phase2.shape[2])).astype('uint8')
for i in range(us_imgs_phase2.shape[0]):
    us_imgs_phase2_crop[i] = us_imgs_phase2[i, gel_mean[i]:int(gel_mean[i]+230), :]

us_imgs = np.vstack((us_imgs_phase1_crop, us_imgs_phase2_crop, us_imgs_extract))

# Get labels
labels_tot = np.squeeze(tumor_or_not['TumorInImage']).astype('int64')
labels_tot = np.append(labels_tot, extract_frames['labels'])

# Load labels and metadata
label_meta = pd.read_csv('Data/table_tot_crop_checked.csv')

# Get patient IDs
pat_ids = np.copy(label_meta['PatientID'].to_numpy())
pat_ids = np.array([item.split('_')[0] for item in pat_ids])
pat_ids = np.append(pat_ids, extract_frames['pat_ids'])

us_img_no = np.squeeze(tumor_or_not['USimageNr']).astype('int64')
us_img_no = np.append(us_img_no, np.arange(226, 226+52))
us_img_no = us_img_no - 1

# Get labels per patient
pat_ids_unique = np.unique(pat_ids)
us_pats = np.empty(shape=labels_tot.shape, dtype='<U5')
labels_stratify = np.zeros(shape=pat_ids_unique.shape)
for count, item in enumerate(pat_ids_unique):
    pat_idx = np.where(pat_ids == item)
    us_no = np.unique(us_img_no[pat_idx])
    print(item)
    us_pats[us_no]=item
    if np.any(labels_tot[us_no] == 1):
        labels_stratify[count]=1


# Split in train and test set
from sklearn.model_selection import train_test_split
train_ids, test_ids, y_train, y_test = train_test_split(pat_ids_unique, labels_stratify,
                                                      test_size=0.2, random_state=3,
                                                      stratify=labels_stratify)


# Select data according to ids
train_us = us_imgs[ids2indices(us_pats, train_ids)]
train_labels = labels_tot[ids2indices(us_pats, train_ids)]
train_pats = us_pats[ids2indices(us_pats, train_ids)]

# Select test data
test_us = us_imgs[ids2indices(us_pats, test_ids)]
test_labels = labels_tot[ids2indices(us_pats, test_ids)]
test_pats = us_pats[ids2indices(us_pats, test_ids)]

''''
AUGMENT EVERYTHING
'''
if run_params['augment'][0]:
    augment_ratio = 4
    augment_factors = np.array([2, 1]) * augment_ratio
    _, counts = np.unique(train_labels, return_counts=True)

    # Pre-allocate array
    pat_ids_augment = np.array([])
    labels_augment = np.array([])
    us_imgs_augment = np.zeros(shape=(augment_factors[0]*counts[0]+augment_factors[1]*counts[1], 230, 344)) # specify size of data array for speed
    sample_loc=0


    # Set random state for reproducable results
    r = np.random.RandomState(22)
    for label_no in range(2):
        labels_idx = np.where(train_labels == label_no)[0]
        factor = augment_factors[label_no]

        # Select data needed for augmentation
        pat_ids_augment_temp = train_pats[labels_idx]
        us_imgs_augment_temp = train_us[labels_idx]
        labels_augment_temp = train_labels[labels_idx]

        for augment_cycle in range(factor):

            # Create random matrix for augmentation strategies
            gamma_numbers = r.uniform(low=0.8, high=1.2, size=pat_ids_augment_temp.shape)
            flip_numbers = r.randint(low=0, high=2, size=pat_ids_augment_temp.shape)
            rot_numbers = r.randint(low=-5, high=5, size=pat_ids_augment_temp.shape)

            for sample_no, sample in enumerate(us_imgs_augment_temp):

                # Flip sample if needed
                if flip_numbers[sample_no] == 1:
                    sample = sample.copy()
                    flip_t = createTechnique("flip", {"flip": flip_numbers[sample_no]})
                    sample = flip_t.apply(sample)

                # Rotate sample
                sample = rot_crop_image(sample, rot_numbers[sample_no])
                sample = cv2.resize(sample, (344, 230))

                # Apply gamma correction
                gamma_t = createTechnique('gamma', {'gamma': gamma_numbers[sample_no]})
                sample = gamma_t.apply(sample)

                # Append sample, pat_id and label to dataset
                labels_augment = np.append(labels_augment, labels_augment_temp[sample_no])
                us_imgs_augment[sample_loc] = sample
                pat_ids_augment = np.append(pat_ids_augment, pat_ids_augment_temp[sample_no])
                sample_loc = sample_loc + 1
                print('Sample {}/{} done of cycle {}'.format(sample_no+1, us_imgs_augment_temp.shape[0], augment_cycle))
            print('Factor {}/{} done of label {}'.format(augment_cycle+1, factor, label_no))
        print('Label done')

    # Add everything together
    train_us = np.append(train_us, us_imgs_augment, axis=0)
    train_labels = np.append(train_labels, labels_augment)
    train_pats = np.append(train_pats, pat_ids_augment)

'''
Normalize
'''
if run_params['normalize'][0]:
    train_us = norm_us(train_us)
    test_us = norm_us(test_us)

'''
CREATE LABELS FOR STRATIFIED K-FOLD CROSS VALIDATION
- Gets tumor and fibrosis labels to equally distribute harder examples
'''

data = {}
data['train_ids'] = train_ids
data['labels_stratify'] = y_train
data['train_us'] = train_us
data['train_labels'] = train_labels
data['train_ids_augment'] = train_pats
data['test_us'] = test_us
data['test_labels'] = test_labels
data['test_ids_augment'] = test_pats
data['test_ids'] = test_ids
data['labels_stratify_test'] = y_test

# Save data as .npy files for DataGenerator use in training
for count, item in enumerate(data['train_us']):
    np.save('C:/Users/s.pruijssers/data_us_complete/sample_{}'.format(count), item)

with open('Data processed/complete_us_cropped2.pkl', 'wb') as file:
    pickle.dump(data, file)