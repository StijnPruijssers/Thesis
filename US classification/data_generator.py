import numpy as np
from tensorflow import keras
from clodsa.techniques.techniqueFactory import createTechnique
from us_augment import rot_crop_image
import cv2
from split_preprocess_augment import norm_us
from tensorflow.keras.utils import to_categorical


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data_idx, labels, batch_size=32, dim=(230, 344), shuffle=True, augment=True, channels=1):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.data_idx = data_idx
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augment = augment
        self.channels = channels

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_idx) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.batch_indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        data_idx_temp = [self.data_idx[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(data_idx_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.batch_indexes = np.arange(len(self.data_idx))
        if self.shuffle == True:
            np.random.shuffle(self.batch_indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        if self.augment:
            # Create random matrix for augmentation strategies
            gamma_numbers = np.random.uniform(low=0.8, high=1.2, size=self.batch_size)
            flip_numbers = np.random.randint(low=0, high=2, size=self.batch_size)
            rot_numbers = np.random.randint(low=-5, high=5, size=self.batch_size)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Load sample
            sample = np.load('C:/Users/s.pruijssers/data_us_complete/sample_{}.npy'.format(ID))

            if not sample.shape == self.dim:
                sample = cv2.resize(sample, self.dim, interpolation=cv2.INTER_CUBIC)

            if self.augment:

                # Flip sample if needed
                if flip_numbers[i] == 1:
                    sample = sample.copy()
                    flip_t = createTechnique("flip", {"flip": flip_numbers[i]})
                    sample = flip_t.apply(sample)

                # Rotate sample
                sample = rot_crop_image(sample, rot_numbers[i])
                sample = cv2.resize(sample, (self.dim[1], self.dim[0]))


                # Apply gamma correction
                gamma_t = createTechnique('gamma', {'gamma': gamma_numbers[i]})
                sample = gamma_t.apply(sample)

            # STore sample
            X[i, ] = sample

            # Store class
            y[i] = self.labels[ID]

        # Normalize samples
        X = norm_us(X).astype('float32')

        if self.channels == 3:
            X = np.tile(X, (3, 1, 1, 1))
            X = np.transpose(X, (1, 0, 2, 3))
        elif self.channels == 1:
            # Add dimension for training
            X = np.expand_dims(X, -1)
        else:
            raise NotImplementedError

        return X, y