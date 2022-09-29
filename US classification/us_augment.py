import os
from scipy.io import loadmat
import numpy as np
from pydicom import dcmread
import pandas as pd
from matplotlib import pyplot as plt
import cv2
import math


def rotate_img(img, angle, center=(0, 0)):
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # rotate the original image
    img_rot = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img_rot


def get_drs_rectangles(label_meta, contactY_val, banana_width, measure_depth, us_width=344):

    rect_pts = np.zeros(shape=(label_meta.shape[0], 4, 2))
    # Return image rectangle coordinates for projective transform
    for sample in range(label_meta.shape[0]):
        if label_meta['US_Location'][sample] == 'left':
            meas_idx = np.int32(np.round(us_width * 0.15))
        elif label_meta['US_Location'][sample] == 'middle':
            meas_idx = np.int32(np.round(us_width * 0.5))
        elif label_meta['US_Location'][sample] == 'right':
            meas_idx = np.int32(np.round(us_width * 0.85))

        # Get coordinates
        rect_pts[sample, :] = np.float32([[meas_idx - banana_width, contactY_val[sample] + measure_depth],
                                          [meas_idx - banana_width, contactY_val[sample]],
                                          [meas_idx + banana_width, contactY_val[sample]],
                                          [meas_idx + banana_width, contactY_val[sample] + measure_depth]])

    return rect_pts


def projective_augmentation(rect_pts, dst_pts, us_imgs, x_dev, y_dev, show=False):
    # Try projective transform, point are in order of
    # left-bottom, left-top, right-top, right-bottom

    # Get height en width of intended rectangle
    rect_width = np.max(dst_pts[:, 0]).astype('int32')
    rect_height = np.max(dst_pts[:, 1]).astype('int32')

    rect_pts = rect_pts.astype('float32')
    dst = np.zeros((us_imgs.shape[0], rect_height, rect_width)).astype('float32')
    for item in range(us_imgs.shape[0]):
        # Generate random points x,y translations for projective transform
        rand_y = np.random.randint(low=-x_dev, high=x_dev, size=4)
        rand_x = np.random.randint(low=-y_dev, high=y_dev, size=4)

        # Clip y_values of top corners to not exceed 15
        rand_y[1:3] = np.clip(rand_y[1:3], -5, y_dev)


        # Randomize shift of corners for projective transform
        src_pts = np.copy(rect_pts[item])
        src_pts[:, 0] = src_pts[:, 0] + rand_x
        src_pts[:, 1] = src_pts[:, 1] + rand_y

        # Clip src_pts so it is contained in the image
        src_pts[:, 0] = np.clip(src_pts[:, 0], 0, us_imgs.shape[2]-1)
        src_pts[:, 1] = np.clip(src_pts[:, 1], 0, us_imgs.shape[1]-1)

        # Get projective transform
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # Get transformed rectangle
        dst[item, :, :] = cv2.warpPerspective(us_imgs[item, :, :], M, (rect_width, rect_height))

        if show:
            number = len(os.listdir('Visualisations/us_augment_vis'))
            fig, axs = plt.subplots(1, 3, figsize=(16, 8))
            axs = axs.ravel()
            axs[0].imshow(us_imgs[item], cmap='gray')
            axs[0].plot(rect_pts[item, :, 0], rect_pts[item, :, 1], 'or', markersize=4)
            axs[0].plot(src_pts[:, 0], src_pts[:, 1], 'og', markersize=4)
            axs[1].imshow(us_imgs[item, int(np.min(rect_pts[item, :, 1])):int(np.max(rect_pts[item, :, 1])),
                          int(np.min(rect_pts[item, :, 0])):int(np.max(rect_pts[item, :, 0]))], cmap='gray')
            axs[2].imshow(dst[item], cmap='gray')
            fig.savefig('us_augment_vis/us_augment_{}.jpg'.format(number))
            plt.close(fig)


    return dst, rect_pts


def us_ratio_augment(labels_ratios, labels_idx, rect_pts, dst_pts, us_imgs, x_dev, y_dev, labels, pat_ids, sample_ids, show=False):

    # Create empty array to append to
    rects_augment = np.empty(shape=(0, int(np.max(dst_pts[:, 1])), int(np.max(dst_pts[:, 0]))))

    # Select rectangles from original images
    us_rect_orig = select_rectangles(rect_pts, us_imgs, dst_pts)

    # Pre-allocate arrays for copying labels and pat_ids to augmented samples
    if labels.ndim == 1:
        labels_augment = np.array([])
    else:
        labels_augment = np.empty((0, 3))

    pat_ids_augment = np.array([])
    sample_ids_augment = np.array([])

    # Augment per class
    for i, item in enumerate(labels_idx):
        augment_factor = labels_ratios[i]
        us_augment_temp = us_imgs[item[0]]
        rects_temp = rect_pts[item[0]]

        # Loop through amount of augmentation needed
        for count, ratio in enumerate(range(0, augment_factor)):
            if ratio == 0:
                augment_data, _ = projective_augmentation(rects_temp, dst_pts, us_augment_temp, x_dev, y_dev, show=show)
                augment_labels = labels[item[0]]
            else:
                augment_data_temp, _ = projective_augmentation(rects_temp, dst_pts, us_augment_temp, x_dev, y_dev, show=show)
                augment_data = np.append(augment_data,
                                         augment_data_temp,
                                         axis=0)
                augment_labels = np.append(augment_labels, labels[item[0]], axis=0)

        pat_ids_augment = np.append(pat_ids_augment, np.tile(pat_ids[item[0]], labels_ratios[i]))
        sample_ids_augment = np.append(sample_ids_augment, np.tile(sample_ids[item[0]], labels_ratios[i]))
        rects_augment = np.append(rects_augment, augment_data, axis=0)
        labels_augment = np.append(labels_augment, augment_labels, axis=0)

    # Add augmented data to original data
    rects_data_tot = np.append(us_rect_orig, rects_augment, axis=0)
    labels_tot = np.append(labels, labels_augment, axis=0)
    pat_ids_tot = np.append(pat_ids, pat_ids_augment)
    sample_ids_tot = np.append(sample_ids, sample_ids_augment)

    return rects_data_tot, labels_tot, pat_ids_tot, sample_ids_tot


def select_rectangles(rect_pts, us_imgs, dst_pts):
    # Select original rectangles
    rect_pts_original = np.copy(rect_pts)
    rect_pts_original = rect_pts_original.astype('int64')
    us_rect_orig = np.empty(shape=(0, int(np.max(dst_pts[:, 1])), int(np.max(dst_pts[:, 0]))))
    for sample_no in range(us_imgs.shape[0]):
        rect_pts_original_temp = rect_pts_original[sample_no]
        us_rect_orig_temp = us_imgs[sample_no, np.min(rect_pts_original_temp[:, 1]): np.max(rect_pts_original_temp[:, 1]),
                                               np.min(rect_pts_original_temp[:, 0]): np.max(rect_pts_original_temp[:, 0])]
        us_rect_orig_temp = np.expand_dims(us_rect_orig_temp, axis=0)
        us_rect_orig = np.append(us_rect_orig, us_rect_orig_temp, axis=0)

    return us_rect_orig


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rot_crop_image(image, angle):
    image_height, image_width = image.shape[0:2]
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(
        image_rotated,
        *largest_rotated_rect(
            image_width,
            image_height,
            math.radians(angle)
        )
    )
    return image_rotated_cropped


if __name__ == '__main__':

    # Load US images
    us_imgs = loadmat('Data/US_imgs_tot.mat')
    us_imgs = us_imgs['US_imgs_tot']

    # Load labels and metadata
    label_meta = pd.read_csv('Data/table_tot_crop.csv')

    # Load GelY_vals for phase 2 measurements, calculated with ExtractGelY.m
    GelY_vals = loadmat('Data/GelY_vals.mat')
    GelY_vals = GelY_vals['GelY_vals']

    # Create array for all Y contact values
    contactY_val = np.full((us_imgs.shape[0]), 15, dtype='int32')  # All phase 1 values are 15
    contactY_val[394::] = GelY_vals  # Replace phase 2 values with calculated GelY_vals

    # Plot US image corresponding to image
    # Get pixel spacing in mm/px, same for both x and y
    pixel_spacing = dcmread('Data/example_DICOM_US')[0x18, 0x6011][0]['PhysicalDeltaX'].value * 10  # in mm/px
    banana_width = np.int32(np.round(2.5 / pixel_spacing))  # Banana width is about 5 mm, so 2.5 per side
    measure_depth = np.int32(np.round(10 / pixel_spacing))  # Maximum measure depth is about 10 mm

    # Get list of rectangles of DRS location with height=measure_depth and width=2*banana_width combined
    # with drs location and depth of gel_layer/contact layer
    rect_pts = get_drs_rectangles(label_meta, contactY_val, banana_width, measure_depth, us_width=us_imgs.shape[-1])

    # Define rectangle shape as destination for the projective transform
    dst_pts = np.float32([[0, measure_depth],
                          [0, 0],
                          [banana_width*2, 0],
                          [banana_width*2, measure_depth]])

    # Transform all rects once
    rects_transformed, rect_pts_transformed = projective_augmentation(rect_pts, dst_pts, us_imgs, 6, 6)

    # Do the first sample thirty times for visual inspection
    rect_pts_test = np.expand_dims(rect_pts[0], 0)
    rect_pts_test = np.repeat(rect_pts_test, 30, axis=0)

    us_img_test = np.expand_dims(us_imgs[0], 0)
    us_img_test = np.repeat(us_img_test, 30, axis=0)

    rects_transformed_test, rect_pts_transformed_test = projective_augmentation(rect_pts_test, dst_pts,
                                                                                us_img_test, 6, 6)

    # Show the set of examples for checking of augmentation
    for item in range(0, 30):
        meas_idx = 52
        fig3, (c0, c1, c2) = plt.subplots(1, 3, figsize=(18, 6))
        c0.imshow(us_imgs[0], cmap='gray')
        c0.set_title('Original image')
        c0.plot(rect_pts[0, :, 0], rect_pts[0, :, 1], 'or', markersize=4)
        c0.plot(rect_pts_transformed_test[item, :, 0], rect_pts_transformed_test[item, :, 1], 'og', markersize=4)
        c0.legend(['original points', 'projective \n transform points'])
        c1.imshow(
            us_imgs[0, contactY_val[0]:contactY_val[0] + measure_depth, meas_idx - banana_width:meas_idx + banana_width],
            cmap='gray')
        c1.set_title('Original selected rectangle')
        c2.imshow(rects_transformed_test[item], cmap='gray')
        c2.set_title('Rectangle after projective transform')
        plt.suptitle('Effect of projective augmentation on sample 0: example {}'.format(item), fontweight='bold')
        plt.savefig('rect_test/test{}'.format(item))
        plt.close()
