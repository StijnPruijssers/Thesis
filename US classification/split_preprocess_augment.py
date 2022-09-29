import copy
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


def interpolate_array1d(fib_data, down_rate=320):
    fib_data_downsample = np.empty(shape=(fib_data.shape[0], down_rate, fib_data.shape[-1]))
    for sample_no in range(fib_data.shape[0]):
        for fiber_no in range(fib_data.shape[-1]):
            interp_fun = interp1d(np.arange(0, fib_data.shape[1]), fib_data[sample_no, :, fiber_no])
            fib_data_downsample[sample_no, :, fiber_no] = np.apply_along_axis(interp_fun, 0,
                                                                              np.linspace(0, fib_data.shape[1] - 1,
                                                                                          num=down_rate))

    return fib_data_downsample


def drs_stretch_noise_slope(fib_data, stretch_delta, offset_delta, slope_delta, std):
    # Pre-allocate array
    augment_data = np.zeros(fib_data.shape)
    for count, sample in enumerate(fib_data):
        stretch_num = np.random.uniform(0 - stretch_delta, 0 + stretch_delta)
        off_num = np.random.uniform(0 - offset_delta, 0 + offset_delta)
        slope_num = np.random.uniform(0 - slope_delta, 0 + slope_delta)

        for fib_no in range(fib_data.shape[-1]):
            slope_fiber = (fib_data[count, -1, fib_no] - fib_data[count, 0, fib_no]) / fib_data.shape[1]
            slope_fiber = slope_fiber * slope_num

            # Augment data in form of augment_data = offset + original_data * stretching + slope
            augment_data[count, :, fib_no] = off_num * std[fib_no] + \
                                             fib_data[count, :, fib_no] * (1 + (stretch_num * std[fib_no])) \
                                             + slope_fiber * np.arange(1, fib_data.shape[1] + 1)

    return augment_data


def split_data(pat_ids, labels, train_size, val_size, test_size, random_state):
    # Define split ratios for train and test/val split
    split1 = 1 - train_size
    split2 = test_size / (val_size + test_size)

    # Create list of indices equal to pat_id size for slicing of arrays
    pat_ids_ix = np.arange(start=0, stop=pat_ids.shape[0], step=1)

    # Split ids in train, val and test set.
    train_ids_idx, val_ids_idx = train_test_split(pat_ids_ix, test_size=split1, shuffle=True,
                                                  random_state=random_state, stratify=labels)

    if test_size != 0:
        val_ids_idx, test_ids_idx = train_test_split(val_ids_idx, test_size=split2, shuffle=True,
                                                     random_state=random_state, stratify=labels[val_ids_idx])
    else:
        test_ids_idx = val_ids_idx

    # Select corresponding ids for indices (idx)
    train_ids = pat_ids[train_ids_idx]
    val_ids = pat_ids[val_ids_idx]
    test_ids = pat_ids[test_ids_idx]

    return train_ids, val_ids, test_ids


def ids2indices_all(pat_ids, train_ids, val_ids, test_ids):
    # Get indices of selected train, test and val ids
    train_idx = np.array([])
    for item in train_ids:
        train_idx = np.append(train_idx, np.where(pat_ids == item)[0])
    train_idx = train_idx.astype('uint64')

    val_idx = np.array([])
    for item in val_ids:
        val_idx = np.append(val_idx, np.where(pat_ids == item)[0])
    val_idx = val_idx.astype('uint64')

    test_idx = np.array([])
    for item in test_ids:
        test_idx = np.append(test_idx, np.where(pat_ids == item)[0])
    test_idx = test_idx.astype('uint64')

    return train_idx, val_idx, test_idx


def ids2indices(pat_ids, selected_ids):
    # Get indices of selected train, test and val ids
    indices = np.array([])
    for item in selected_ids:
        indices = np.append(indices, np.where(pat_ids == item)[0])
    indices = indices.astype('uint64')
    return indices


def drs_ratio_augment(fib_data, labels, augment_factor, stretch_delta, offset_delta, slope_delta, std):
    # Data Augmentation for drs data
    for ratio in range(0, augment_factor):
        if ratio == 0:
            augment_data = drs_stretch_noise_slope(fib_data, stretch_delta, offset_delta, slope_delta, std)
            augment_labels = labels
        else:
            augment_data = np.append(augment_data,
                                     drs_stretch_noise_slope(fib_data, stretch_delta, offset_delta, slope_delta, std),
                                     axis=0)
            augment_labels = np.append(augment_labels, labels, axis=0)

    return augment_data, augment_labels


def get_combination_stratify_labels(labels_all, pat_ids, k_fold=True, simple=False):
    # Get unique combinations present in the first two layers
    unique_rows, counts = np.unique(labels_all[:, 0:2], axis=0, return_counts=True)

    # Get which patients correspond to the combinations
    comb_pts = []
    for item in unique_rows:
        indices = np.where(np.all(np.equal(item, labels_all[:, 0:2]), axis=1))[0]
        comb_pts.append(np.unique(pat_ids[indices]))

    # Get a label for every patient based on the first to layer for stratified splitting
    combs_per_pat = []
    counts_per_pat = []
    max_count_of_pat = []
    pt_stratify_comb_max = []

    pat_ids_unique = np.unique(pat_ids)  # All unique pat_ids combinations
    for i, pat_id in enumerate(pat_ids_unique):  # For every patient

        # Get all layer combinations in first two layers and how often they occur
        combs, counts_id_combs = np.unique(labels_all[np.where(pat_ids == pat_id), 0:2], axis=1, return_counts=True)
        combs_per_pat.append(combs)
        counts_per_pat.append(counts_id_combs)

        # Get the index of the most frequently occurring combination
        max_count_of_pat.append(np.max(counts_id_combs) / np.sum(counts_id_combs))
        max_idx = np.argwhere(counts_per_pat[i] == np.max(counts_per_pat[i]))

        # If there are multiple combinations which occur the most often,
        # choose the one that occurs less in the complete dataset.
        pt_comb_counts = np.array([])

        if max_idx.shape[0] > 1:  # If there is more than one max
            for item in combs[0, max_idx]:  # Get counts how often the combination occurs in the complete dataset
                pt_comb_counts = np.append(pt_comb_counts,
                                           counts[np.where(np.all(np.equal(unique_rows, item), axis=1))])

            # Get minimal maximum combination and assign as stratifying label for the patient
            min_max_idx = np.argwhere(pt_comb_counts == np.min(pt_comb_counts))
            pt_stratify_comb_max.append(combs[0, min_max_idx][0][0])

        else:  # If there is only one max, assign that combination for stratified splitting
            pt_stratify_comb_max.append(combs[0, max_idx][0][0])

    # Get unique combinations and patients from max_min label assignment
    pt_stratify_comb_max = np.array(pt_stratify_comb_max)
    unique_combs_max = np.unique(pt_stratify_comb_max, axis=0)

    # Get all patients that have been assigned to one of the unique combinations
    pat_stratify_max = []
    for item in unique_combs_max:
        pat_stratify_max.append(pat_ids_unique[np.where(np.all(np.equal(item, pt_stratify_comb_max), axis=1))])

    # Put data into dataframe for easy viewing
    df_combs_final_max = pd.DataFrame(data=unique_combs_max, columns=['Layer 1', 'Layer 2'])
    df_combs_final_max['PatientIDs'] = pat_stratify_max

    if not simple:
        # Remove combination that only occurs once
        df_combs_final_max = df_combs_final_max.drop(7, axis=0)

        if not k_fold:
            fibrosis_comb_total = np.append(df_combs_final_max['PatientIDs'][8], (df_combs_final_max['PatientIDs'][9],
                                                                                  df_combs_final_max['PatientIDs'][11]))
            df_combs_final_max['PatientIDs'][8] = fibrosis_comb_total
            df_combs_final_max = df_combs_final_max.drop(9, axis=0)
            df_combs_final_max = df_combs_final_max.drop(11, axis=0)

    # Get final stratified labels with corresponding patient IDs
    # Turns combinations to singular integer class
    labels_stratify = np.array([])
    pat_stratify = []
    for i in range(df_combs_final_max.shape[0]):
        for j in df_combs_final_max['PatientIDs'].iloc[i]:
            labels_stratify = np.append(labels_stratify, i)
            pat_stratify.append(j)

    pat_stratify = np.array(pat_stratify)
    labels_stratify = labels_stratify.astype('int64')

    return labels_stratify, pat_stratify, df_combs_final_max


def get_relative_stratify_labels(meta_data, pat_ids):
    # Set NaN thicknesses to 0
    nan_layer2_idx = meta_data[meta_data['PA_ThickLayer2'].isnull()].index
    nan_layer3_idx = meta_data[meta_data['PA_ThickLayer3'].isnull()].index
    meta_data.loc[nan_layer2_idx, 'PA_ThickLayer2'] = 0
    meta_data.loc[nan_layer3_idx, 'PA_ThickLayer3'] = 0

    # Select tumor occurrences in first layer and set to tumor label if tumor presence is over 0.5 mm
    tumor1 = meta_data[(meta_data['PA_LabelLayer1'] == 2) & (meta_data['PA_ThickLayer1'] >= 0.5)]
    meta_data.loc[tumor1.index, 'RelLabel'] = 2

    # Select tumor occurrences in second layer and get occurrences where there is less than 1.5 mm of top layer
    tumor2 = meta_data[(meta_data['PA_LabelLayer2'] == 2) & (meta_data['PA_ThickLayer1'] < 1.5)]
    meta_data.loc[tumor2.index, 'RelLabel'] = 2

    # Select occurrences in third layer and select samples where top layers are less than 1.5 mm
    tumor3 = meta_data[
        (meta_data['PA_LabelLayer3'] == 2) & ((meta_data['PA_ThickLayer1'] + meta_data['PA_ThickLayer2']) < 1.5)]
    meta_data.loc[tumor3.index, 'RelLabel'] = 2

    # Get healthy training labels
    healthy = meta_data[meta_data['RelLabel'] == 0]
    healthy1 = healthy[healthy['PA_ThickLayer1'] >= 1]
    meta_data.loc[healthy1.index, 'RelLabel'] = healthy1['PA_LabelLayer1']

    healthy2 = healthy[healthy['PA_ThickLayer1'] < 1]
    meta_data.loc[healthy2.index, 'RelLabel'] = healthy2['PA_LabelLayer2']

    # Get unique prefix and to get per patient labels for stratified splitting
    pat_ids_unique = np.unique(pat_ids)
    labels_stratify = np.zeros(shape=pat_ids_unique.shape)
    tot_vals, tot_counts = np.unique(meta_data['RelLabel'].to_numpy(), return_counts=True)

    for count, pat_id_unique in enumerate(pat_ids_unique):
        pat_labels, pat_counts = np.unique(meta_data[meta_data['PatientID'] == pat_id_unique]['RelLabel'],
                                           return_counts=True)
        max_idx = (pat_counts == np.max(pat_counts))
        if len(max_idx) == 1:
            labels_stratify[count] = pat_labels[np.argmax(pat_counts)]
        else:
            max_labels = pat_labels[max_idx].astype('int64')
            labels_stratify[count] = np.where(tot_counts == np.min(tot_counts[max_labels]))[0]

    # Redefine pat_ids_unique as pat ids used for stratification
    pat_stratify = pat_ids_unique

    return labels_stratify, pat_stratify, meta_data


def check_dataset_log(params):
    # Check if there is a log file present and if current dataset is already made
    if not os.path.isfile('dataset_creation_log.xlsx'):

        # If it does not exist, create one
        params['dataset_name'][0] = params['dataset_name'][0] + '0'  # Create first dataset name
        dataset_creation_log = pd.DataFrame.from_dict(data=params)

    else:
        # If it exists, load and check params
        dataset_creation_log = pd.read_excel('dataset_creation_log.xlsx', index_col=[0])

        # Add new number after dataset
        params['dataset_name'][0] = params['dataset_name'][0] + str(
            dataset_creation_log.shape[0])  # Set dataset name
        print('Adding new dataset')

    return dataset_creation_log


def save_dataset_log(params, dataset_creation_log):
    if not os.path.isfile('dataset_creation_log.xlsx'):
        dataset_creation_log.to_excel('dataset_creation_log.xlsx')
        print('Created dataset {} and log for dataset creation'.format(params['dataset_name'][0] + '.pkl'))

    else:
        # If succeeded add to data log and write to file
        dataset_creation_log_temp = pd.DataFrame.from_dict(data=params)
        dataset_creation_log = dataset_creation_log.append(dataset_creation_log_temp, ignore_index=True)
        dataset_creation_log.to_excel('dataset_creation_log.xlsx')
        print('New data set created as {}'.format(params['dataset_name'][0] + '.pkl'))


def get_label_ratios(labels_all, ratio_cap):
    # Get counts of combinations in test and test set

    if labels_all.ndim == 1:
        labels, labels_counts = np.unique(labels_all, axis=0, return_counts=True)
    else:
        labels, labels_counts = np.unique(labels_all[:, 0:2], axis=0, return_counts=True)

    labels_ratios = 1 / (labels_counts / np.max(labels_counts))
    labels_ratios[labels_ratios > ratio_cap] = ratio_cap
    labels_ratios = labels_ratios.astype('int64')

    # Get all indices where labels are present
    labels_idx = []
    if labels_all.ndim == 1:
        for item in labels:
            labels_idx.append(np.where(labels_all == item))
    else:
        for item in labels:
            labels_idx.append(np.where(np.all(np.equal(labels_all[:, 0:2], item), axis=1)))

    return labels_ratios, labels_idx


def drs_augment(labels, fib_data, pat_ids, params, labels_ratios, labels_idx, sample_ids=None):
    # Calculate standard deviation per fibre
    std = np.array([np.std(item) for item in fib_data.transpose(2, 1, 0)])

    # Augment data
    fib_augment = np.empty((0, fib_data.shape[1], fib_data.shape[2]))

    # Pre-allocate arrays for copying labels and pat_ids to augmented samples
    if labels.ndim == 1:
        labels_augment = np.array([])
    else:
        labels_augment = np.empty((0, 3))

    pat_ids_augment = np.array([])
    sample_ids_augment = np.array([])

    for i, item in enumerate(labels_idx):
        augment_data, augment_labels = drs_ratio_augment(fib_data=fib_data[item[0]], labels=labels[item[0]],
                                                         augment_factor=labels_ratios[i],
                                                         stretch_delta=params['drs_stretch_delta'],
                                                         offset_delta=params['drs_offset_delta'],
                                                         slope_delta=params['drs_slope_delta'], std=std)

        pat_ids_augment = np.append(pat_ids_augment, np.tile(pat_ids[item[0]], labels_ratios[i]))
        sample_ids_augment = np.append(sample_ids_augment, np.tile(sample_ids[item[0]], labels_ratios[i]))
        fib_augment = np.vstack((fib_augment, augment_data))
        labels_augment = np.append(labels_augment, augment_labels, axis=0)

    fib_data_tot = np.vstack((fib_data, fib_augment))
    labels_tot = np.append(labels, labels_augment, axis=0)
    pat_ids_tot = np.append(pat_ids, pat_ids_augment)
    sample_ids_tot = np.append(sample_ids, sample_ids_augment)

    return fib_data_tot, labels_tot, pat_ids_tot, sample_ids_tot


def norm_us(us_imgs):
    us_norm = np.zeros(shape=us_imgs.shape)
    for count, item in enumerate(us_imgs):
        us_norm[count] = item / item.max()

    return us_norm


def drs_min_max_norm(fib_data):
    fib_data = np.copy(fib_data)
    for sample_no in range(fib_data.shape[0]):
        fib_data[sample_no] = (fib_data[sample_no] - np.min(fib_data[sample_no])) / (
            np.max(fib_data[sample_no]) - np.min(fib_data[sample_no]))

    return fib_data


def snv(input_data):
    """
        :snv: A correction technique which is done on each
        individual spectrum, a reference spectrum is not
        required
        :param input_data: Array of spectral data
        :type input_data: DataFrame

        :returns: data_snv (ndarray): Scatter corrected spectra
    """

    input_data = np.asarray(input_data)

    # Define a new array and populate it with the corrected data
    data_snv = np.zeros(shape=input_data.shape)
    for sample_no in range(input_data.shape[0]):
        for fib_no in range(input_data.shape[-1]):
            data_snv[sample_no, :, fib_no] = (input_data[sample_no, :, fib_no] - np.mean(input_data[sample_no, :, fib_no]))/ \
                                             np.std(input_data[sample_no, :, fib_no])

    return data_snv


def norm_800(input_data):
    input_data = np.asarray(input_data)
    norm_data = np.zeros(shape=input_data.shape)
    for sample_no in range(input_data.shape[0]):
        for fib_no in range(input_data.shape[-1]):
            norm_data[sample_no, :, fib_no] = input_data[sample_no, :, fib_no] / input_data[sample_no, 200, fib_no]

    return norm_data


def magnitude_norm(input_data):
    input_data = np.asarray(input_data)
    mag_norm = np.zeros(shape=input_data.shape)

    for sample_no in range(input_data.shape[0]):
        for fib_no in range(input_data.shape[-1]):
            mag_norm[sample_no, :, fib_no] = input_data[sample_no, :, fib_no] / np.max(input_data[sample_no, :, fib_no])

    return mag_norm




