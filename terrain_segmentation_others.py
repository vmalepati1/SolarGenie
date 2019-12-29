import json
import os
from time import time

import keras
import numpy as np
import segmentation_models as sm
from PIL import Image
# Number of classes (including background)
from keras.utils import Sequence
from skimage.draw import polygon
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, image_path, class_ids,
                 to_fit=True, batch_size=32, dim=(256, 256), mask_scale_factor=1.0,
                 n_channels=1, n_classes=10, shuffle=True):
        """Initialization
        :param image_path: path to images location
        :param class_ids: dictionary of each class and its respective id (index)
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param mask_scale_factor: scale factor for mask polygon points
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.image_path = image_path
        self.class_ids = class_ids
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.mask_scale_factor = mask_scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.file_names = os.listdir(self.image_path)  # List of training image file names
        self.file_names.remove('via_region_data.json')
        self.annotations = json.load(open(os.path.join(self.image_path, "via_region_data.json"))).values()

        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.file_names) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        file_names_temp = [self.file_names[k] for k in indexes]

        # Generate data
        X = self._generate_X(file_names_temp)

        if self.to_fit:
            y = self._generate_y(file_names_temp)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.file_names))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, list_file_names):
        """Generates data containing batch_size images
        :param list_file_names: list of image filenames to load
        :return: batch of images
        """
        # Initialization
        X = np.zeros((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, filepath in enumerate(list_file_names):
            # Store normalized sample
            orig = Image.open(os.path.join(self.image_path, filepath))
            new = orig.resize(self.dim)
            X[i,] = np.asarray(new,'float')/255.0

        return X

    def _generate_y(self, list_file_names):
        """Generates data containing batch_size masks
        :param list_file_names: list of image filenames to load
        :return: batch of masks
        """
        y = np.zeros((self.batch_size, *self.dim, self.n_classes), dtype=int)

        # Load annotations
        # VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }

        # Generate data
        for i, filename in enumerate(list_file_names):
            # Store sample
            try:
              a = next(k for k in self.annotations if k['filename'] == filename)
            except:
              # Image had no regions
              continue

            polygons = [r for r in a['regions'].values()]

            for poly in polygons:
                xp = [int(x * self.mask_scale_factor) for x in poly['shape_attributes']['all_points_x']]
                yp = [int(y * self.mask_scale_factor) for y in poly['shape_attributes']['all_points_y']]
                rr, cc = polygon(xp, yp)
                output_layer_ix = self.class_ids[poly['region_attributes']['building']]
                y[i, rr, cc, output_layer_ix] = 1

        return y


NUM_CLASSES = 1 + 16 + 2 + 1
IMAGE_DIM = 64

TRAINING_IMAGE_DIR = 'deeproof-release/data/final-dataset/train'
VAL_IMAGE_DIR = 'deeproof-release/data/final-dataset/val'

class_ids_dict = {
    "BG": 0,
    "flat": 1,
    "dome": 2,
    "N": 3,
    "NNE": 4,
    "NE": 5,
    "ENE": 6,
    "E": 7,
    "ESE": 8,
    "SE": 9,
    "SSE": 10,
    "S": 11,
    "SSW": 12,
    "SW": 13,
    "WSW": 14,
    "W": 15,
    "WNW": 16,
    "NW": 17,
    "NNW": 18,
    "tree": 19,
}


def get_num_files_in_dir(dir):
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


NUMBER_OF_TRAINING_IMAGES = get_num_files_in_dir(TRAINING_IMAGE_DIR) - 1
NUMBER_OF_VAL_IMAGES = get_num_files_in_dir(VAL_IMAGE_DIR) - 1
print(NUMBER_OF_TRAINING_IMAGES)
print(NUMBER_OF_VAL_IMAGES)
BATCH_SIZE = 32

models = {
    # name, model instance
    'unet_resnet50': sm.Unet('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'unet_resnet101': sm.Unet('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet50': sm.FPN('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet101': sm.FPN('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
}

for name, model in models.items():
    print('Compiling ' + name)
    model.compile(
        'SGD',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir="models/others_logs/{}".format(time()),
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint('models/' + name +'_weights.latest.h5',
                                        verbose=1, save_weights_only=True),
        keras.callbacks.ModelCheckpoint('models/' + name + '_weights.best.h5',
                                        verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
    ]

    print('Creating generators')

    training_generator = DataGenerator(TRAINING_IMAGE_DIR, class_ids_dict, dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM/512,
                                        n_channels=3,
                                        n_classes=NUM_CLASSES)
    validation_generator = DataGenerator(VAL_IMAGE_DIR, class_ids_dict, dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM/512,
                                          n_channels=3,
                                          n_classes=NUM_CLASSES)

    print('Training ' + name)
    model.fit_generator(
        training_generator,
        verbose=1,
        epochs=40,
        steps_per_epoch=NUMBER_OF_TRAINING_IMAGES // BATCH_SIZE,
        callbacks=callbacks,
        validation_data=validation_generator,
        validation_steps=NUMBER_OF_VAL_IMAGES // BATCH_SIZE,
    )
