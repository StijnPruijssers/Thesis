import cv2
from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np


def zero_crossing(image):
    z_c_image = np.zeros(image.shape)

    # For each pixel, count the number of positive
    # and negative pixels in the neighborhood

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            negative_count = 0
            positive_count = 0
            neighbour = [image[i + 1, j - 1], image[i + 1, j], image[i + 1, j + 1], image[i, j - 1], image[i, j + 1],
                         image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1]]
            d = max(neighbour)
            e = min(neighbour)
            for h in neighbour:
                if h > 0:
                    positive_count += 1
                elif h < 0:
                    negative_count += 1

            # If both negative and positive values exist in
            # the pixel neighborhood, then that pixel is a
            # potential zero crossing

            z_c = ((negative_count > 0) and (positive_count > 0))

            # Change the pixel value with the maximum neighborhood
            # difference with the pixel

            if z_c:
                if image[i, j] > 0:
                    z_c_image[i, j] = image[i, j] + np.abs(e)
                elif image[i, j] < 0:
                    z_c_image[i, j] = np.abs(image[i, j]) + d

    # Normalize and change datatype to 'uint8' (optional)
    z_c_norm = z_c_image / z_c_image.max() * 255
    z_c_image = np.uint8(z_c_norm)

    return z_c_image



# Load US images
US_imgs = loadmat('Data/US_imgs_tot.mat')
US_imgs = US_imgs['US_imgs_tot']

phase2_begin = 394
blur_size = 1
ksize_sobel = 7

# Edge detection + saving
for img_no in range(phase2_begin, US_imgs.shape[0]-65):
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(16, 8))
    img = US_imgs[img_no, :, :]
    img = cv2.GaussianBlur(img, (1, 1), 5)
    sobely = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=ksize_sobel)  # Sobel Edge Detection on the Y axis
    sobely2 = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=2, ksize=ksize_sobel)

    a0.imshow(US_imgs[img_no, :, :], cmap='gray')
    a0.axis('off')
    a0.set_title('Original US image')

    a1.imshow(sobely, cmap='gray')
    a1.axis('off')
    a1.set_title('Sobel y: ksize = {}'.format(ksize_sobel))

    a2.imshow(zero_crossing(sobely2), cmap='gray')
    a2.axis('off')
    a2.set_title('LoG in y direction: ksize = {}'.format(ksize_sobel))

    fig.savefig('US_edge/US_edge_entry{}'.format(img_no))
    print('At iteration {}'.format(img_no))
    plt.close(fig)
