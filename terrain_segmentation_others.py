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
import matplotlib.pyplot as plt
from matplotlib.image import imread

from matplotlib.pyplot import figure
import albumentations as A

figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')


class DataGenerator(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """

    def __init__(self, image_path, classes, backbone,
                 to_fit=True, batch_size=32, dim=(256, 256), mask_scale_factor=1.0,
                 shuffle=True):
        """Initialization
        :param image_path: path to images location
        :param class: list of class names
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param mask_scale_factor: scale factor for mask polygon points
        :param shuffle: True to shuffle label indexes after every epoch
        """
        self.image_path = image_path
        self.classes = classes
        self.preprocess_input = self._get_preprocessing(sm.get_preprocessing(backbone))
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.mask_scale_factor = mask_scale_factor
        self.n_classes = len(self.classes)
        self.shuffle = shuffle
        self.file_names = os.listdir(self.image_path)  # List of training image file names
        self.file_names.remove('via_region_data.json')
        self.annotations = json.load(open(os.path.join(self.image_path, "via_region_data.json"))).values()
        self.class_values = [self.classes.index(cls) for cls in self.classes]

        self.on_epoch_end()

    def _get_preprocessing(self, preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)

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
        X = self.preprocess_input(self._generate_X(file_names_temp))

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
        X = []

        print(list_file_names)

        # Generate data
        for i, filepath in enumerate(list_file_names):
            # Store normalized sample
            orig = Image.open(os.path.join(self.image_path, filepath))
            new = orig.resize(self.dim)
            new_rgb = new.convert('RGB')
            image = np.asarray(new_rgb, 'int')
            sample = self.preprocess_input(image=image)
            X.append(sample['image'])

        return np.array(X)

    def _generate_y(self, list_file_names):
        """Generates data containing batch_size masks
        :param list_file_names: list of image filenames to load
        :return: batch of masks
        """
        y = []

        print(list_file_names)

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
            a = next(k for k in self.annotations if k['filename'] == filename)

            mask = np.zeros((*self.dim, self.n_classes), dtype=float)

            if 'regions' in a:
                polygons = [r for r in a['regions'].values()]

                for poly in polygons:
                    xp = [int(x * self.mask_scale_factor) for x in poly['shape_attributes']['all_points_x']]
                    yp = [int(y * self.mask_scale_factor) for y in poly['shape_attributes']['all_points_y']]
                    rr, cc = polygon(xp, yp)
                    class_id = self.classes.index(poly['region_attributes']['building'])
                    mask[rr, cc, class_id] = class_id

            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)

            y.append(mask)

        return np.array(y)


TRAINING_IMAGE_DIR = 'deeproof-release/data/final-dataset/train'
VAL_IMAGE_DIR = 'deeproof-release/data/final-dataset/val'

classes = [
    "flat",
    "dome",
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
    "tree",
]


def get_num_files_in_dir(dir):
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


NUMBER_OF_TRAINING_IMAGES = get_num_files_in_dir(TRAINING_IMAGE_DIR) - 1
NUMBER_OF_VAL_IMAGES = get_num_files_in_dir(VAL_IMAGE_DIR) - 1
print(NUMBER_OF_TRAINING_IMAGES)
print(NUMBER_OF_VAL_IMAGES)
IMAGE_DIM = 512
BATCH_SIZE = 2
LR = 0.0001

n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

models = {
    # name, model instance
    'fpn_resnet101': sm.FPN('resnet101', classes=n_classes, encoder_weights='imagenet', activation=activation),
    'unet_resnet101': sm.Unet('resnet101', classes=n_classes, encoder_weights='imagenet', activation=activation),
    'fpn_resnet50': sm.FPN('resnet50', classes=n_classes, encoder_weights='imagenet', activation=activation),
    'unet_resnet50': sm.Unet('resnet50', classes=n_classes, encoder_weights='imagenet', activation=activation),
}

for name, model in models.items():
    # define optomizer
    optim = keras.optimizers.Adam(LR)

    # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
    # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
    class_weights = np.array(
        [2.36465827, 0.78821942, 0.78821942, 1.01760836, 1.00554493, 0.88297515,
         0.99376417, 1.01682135, 1.00592961, 0.88327175, 0.99451589, 1.01682135,
         1.00516055, 0.88297515, 0.99413989, 1.0172147, 1.00516055, 0.88297515,
         0.99376417, 1.7390873]
    )

    dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
    focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)

    # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
    # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

    metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

    # compile keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir="/content/gdrive/My Drive/SolarGenie/models/others_logs/{}".format(time()),
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint('/content/gdrive/My Drive/SolarGenie/models/' + name + '_weights.latest.h5',
                                        verbose=1, save_weights_only=True),
        keras.callbacks.ModelCheckpoint('/content/gdrive/My Drive/SolarGenie/models/' + name + '_weights.best.h5',
                                        verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
    ]

    print('Creating generators')

    training_generator = DataGenerator(TRAINING_IMAGE_DIR, classes, 'resnet101', batch_size=BATCH_SIZE,
                                       dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM / 512)
    validation_generator = DataGenerator(VAL_IMAGE_DIR, classes, 'resnet101', batch_size=BATCH_SIZE,
                                         dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM / 512)

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
