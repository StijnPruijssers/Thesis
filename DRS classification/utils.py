from matplotlib import pyplot as plt
import numpy as np
import os
from tensorflow.keras.utils import to_categorical
import seaborn as sb
from scipy.ndimage import distance_transform_edt
from tensorflow.keras.utils import to_categorical
from cv2 import resize, INTER_CUBIC
from sklearn.metrics import matthews_corrcoef, recall_score, precision_score, confusion_matrix


def maybe_mkdir_p(directory):
    """
     Maybe makes a directory, checks if already it exists

     :param directory: directory to be creatd
     """

    directory = os.path.abspath(directory)
    splits = directory.split(os.sep)[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def remove_nans(labels):
    # Remove NaNs from labels and set to last known label
    for layer_no in range(1, 3):  # For the last two layer labels
        for label_no in range(labels.shape[0]):
            if np.isnan(labels[label_no, layer_no]):
                labels[label_no, layer_no] = labels[label_no, layer_no - 1]

    return labels


def get_labels(label_meta, method='categorical'):
    if method == 'categorical':

        # Remove NaNs and replace with previous
        labels_format = remove_nans(label_meta)

        # Convert to categorical labels
        labels_format = to_categorical(labels_format, num_classes=4)

        # Transpose dimensions for easy indexing of layers
        labels_format = labels_format.transpose((1, 0, 2))

    elif method == 'binary':

        # Get all binary tumor label
        labels_format = label_meta['TumorLabel'].to_numpy()

    else:
        raise NameError('Define label formatting method')

    return labels_format


def vis_conf_matrix(conf_matrix, dir_name, title):
    sb.heatmap(conf_matrix, annot=True,
               xticklabels=['fat', 'muscle', 'tumor', 'fibrosis'],
               yticklabels=['fat', 'muscle', 'tumor', 'fibrosis'])

    plt.title('Absolute values {}'.format(title))
    plt.xlabel('Pred', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.savefig(dir_name + title + '_absolute' + '.png')
    plt.close()

    conf_recall = np.copy(conf_matrix)
    conf_recall = conf_recall.astype('float64')
    for row_no in range(0, conf_recall.shape[0]):
        conf_recall[row_no, :] = conf_recall[row_no, :] / (np.sum(conf_recall[row_no, :]))

    sb.heatmap(conf_recall, annot=True,
               xticklabels=['fat', 'muscle', 'tumor', 'fibrosis'],
               yticklabels=['fat', 'muscle', 'tumor', 'fibrosis'])

    plt.title('Recall values {}'.format(title))
    plt.xlabel('Pred', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.savefig(dir_name + title + '_recall' + '.png')
    plt.close()

    conf_precision = np.copy(conf_matrix)
    conf_precision = conf_precision.astype('float64')
    for col_no in range(0, conf_precision.shape[0]):
        conf_precision[:, col_no] = conf_precision[:, col_no] / (np.sum(conf_precision[:, col_no]))

    sb.heatmap(conf_precision, annot=True,
               xticklabels=['fat', 'muscle', 'tumor', 'fibrosis'],
               yticklabels=['fat', 'muscle', 'tumor', 'fibrosis'])

    plt.title('Precision values {}'.format(title))
    plt.xlabel('Pred', fontweight='bold')
    plt.ylabel('True', fontweight='bold')
    plt.savefig(dir_name + title + '_precision' + '.png')
    plt.close()


def create_weight_matrix(label_maps, mode=None, class_weights=None):
    if not mode:
        return None

    class_matrix = np.zeros(shape=label_maps.shape, dtype='float32')
    gradient_matrix = np.ones(shape=label_maps.shape, dtype='float32')

    if mode == 'class' or mode == 'all':
        class_matrix = np.copy(label_maps).astype('float32')
        class_matrix[class_matrix == 0] = class_weights[0]
        class_matrix[class_matrix == 1] = class_weights[1]

    if mode == 'gradient' or mode == 'all':
        for count, (label_map, gradient_slice) in enumerate(zip(label_maps, gradient_matrix)):

            if len(np.unique(label_map)) == 1:
                continue

            if len(np.unique(label_map)) == 2:
                label_map = np.copy(label_map).astype('float32')
                row, col = np.where(label_map == 1)
                range_row = row.max() - row.min()
                row_norm = (np.abs(row - row.max()) / range_row) ** 2

                for pixel_no in range(len(row)):
                    gradient_slice[row[pixel_no], col[pixel_no]] = row_norm[pixel_no]

    if mode == 'all':
        weight_matrix = np.multiply(class_matrix, gradient_matrix)
        weight_matrix[weight_matrix < class_weights[0]] = class_weights[0]
    if mode == 'class':
        weight_matrix = class_matrix
    if mode == 'gradient':
        weight_matrix = gradient_matrix

    weight_matrix = np.reshape(weight_matrix, newshape=(label_maps.shape[0], label_maps.shape[1] * label_maps.shape[2]))

    return weight_matrix


def format_us_data(us_imgs, labels, mode='scratch'):
    # If model is pre-trained from Gomez-Flores et al. (2020)
    if mode == 'transfer':
        us_imgs = np.tile(us_imgs, (3, 1, 1, 1))  # Repeat over 3 channels
        us_imgs = np.transpose(us_imgs, (1, 0, 2, 3))  # Put channels first due to import from matlab

        # Turn to one-hot encoding
        labels = to_categorical(labels)

    # If model is trained from scratch
    elif mode == 'scratch':

        # Add dimension for training
        us_imgs = np.expand_dims(us_imgs, -1)
        labels = np.expand_dims(labels, -1)
        labels = to_categorical(labels)

    else:
        raise NotImplementedError

    return us_imgs, labels


def resize_data(imgs, labels, new_size=(128, 128)):
    imgs_resize = np.zeros(shape=(imgs.shape[0], new_size[0], new_size[1]), dtype='uint8')
    labels_resize = np.zeros(shape=(labels.shape[0], new_size[0], new_size[1]), dtype='uint8')

    for count, (img, label_map) in enumerate(zip(imgs, labels)):
        imgs_resize[count] = resize(img, dsize=(new_size[0], new_size[1]), interpolation=INTER_CUBIC)
        labels_resize[count] = resize(label_map, dsize=(new_size[0], new_size[1]), interpolation=INTER_CUBIC)

    return imgs_resize, labels_resize


def drs_calc_loss_weights(fib_data, label_meta):
    intensities = np.mean(fib_data, axis=(0, 1))
    intensities = intensities / intensities[0]

    # Get layer thicknesses
    lay1_thick = label_meta['PA_ThickLayer1'].to_numpy()
    lay2_thick = label_meta['PA_ThickLayer2'].to_numpy()
    lay3_thick = label_meta['PA_ThickLayer3'].to_numpy()

    # Set NaN values as 0
    lay1_thick[np.isnan(lay1_thick)] = 0
    lay2_thick[np.isnan(lay2_thick)] = 0
    lay3_thick[np.isnan(lay3_thick)] = 0

    # Adjust layer thicknesses to sampled thicknesses up to 10 mm
    lay1_thick[lay1_thick > 10] = 10

    for i in range(lay2_thick.shape[0]):
        if (lay2_thick[i] + lay1_thick[i]) > 10:
            lay2_thick[i] = 10 - lay1_thick[i]

    for i in range(lay3_thick.shape[0]):
        if (lay3_thick[i] + lay2_thick[i] + lay1_thick[i]) > 10:
            lay3_thick[i] = 10 - lay1_thick[i] - lay2_thick[i]

    lay1_thick[lay1_thick == 0] = np.nan
    lay2_thick[lay2_thick == 0] = np.nan
    lay3_thick[lay3_thick == 0] = np.nan

    # Calculate means without nans
    lay1_mean = np.nanmean(lay1_thick)
    lay2_mean = np.nanmean(lay2_thick)
    lay3_mean = np.nanmean(lay3_thick)

    lay1_std = np.nanstd(lay1_thick)
    lay2_std = np.nanstd(lay2_thick)
    lay3_std = np.nanstd(lay3_thick)

    # Get min/max depths layers most likely covered by the layer
    lay1_depth = np.array([0, lay1_mean + lay1_std])
    lay2_depth = np.array([(lay1_mean - lay1_std), (lay1_mean - lay1_std) + lay2_mean + lay2_std])
    lay3_depth = np.array([(lay1_mean - lay1_std) + (lay2_mean - lay2_std), ((lay1_mean - lay1_std) + (lay2_mean - lay2_std))
                           + (lay2_mean + lay2_std) + (lay3_mean + lay3_std)])

    # Select fibers that are most likely to contribute to the layers
    fiber_array = [0, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]
    lay1_fibers = np.array([fiber_array[round(lay1_depth[0])], fiber_array[round(lay1_depth[1])]])
    lay2_fibers = np.array([fiber_array[round(lay2_depth[0])], fiber_array[round(lay2_depth[1])]])
    lay3_fibers = np.array([fiber_array[round(lay3_depth[0])], fiber_array[round(lay3_depth[1])]])

    # Add relative intensities together of fibers corresponding to layers for the loss weights
    loss_weight1 = np.sum(intensities[lay1_fibers[0]:lay1_fibers[1]])
    loss_weight2 = np.sum(intensities[lay2_fibers[0]:lay2_fibers[1]])
    loss_weight3 = np.sum(intensities[lay3_fibers[0]:lay3_fibers[1]])

    return np.array([loss_weight1, loss_weight2, loss_weight3])


def calc_metrics(y_true, y_pred, labels=4):

    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_true = y_true.astype('int64')
        y_pred = np.round(np.squeeze(y_pred)).astype('int64')

    confmat = confusion_matrix(y_true, y_pred, labels=labels)

    tp = np.diag(confmat)
    fn = np.sum(confmat, axis=1) - tp
    fp = np.sum(confmat, axis=0) - tp
    tn = np.full(confmat.shape[0], np.sum(confmat)) - (tp + fn + fp)

    specificity = tn / (tn + fp)

    labels = list(range(confmat.shape[0]))

    recall = recall_score(y_true, y_pred, average=None, labels=labels)
    precision = precision_score(y_true, y_pred, average=None, labels=labels)
    mcc = matthews_corrcoef(y_true, y_pred)

    return specificity, recall, precision, mcc, confmat

