import json
import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
from PIL import Image
# Number of classes (including background)
from keras.utils import Sequence
from matplotlib.pyplot import figure
from skimage.draw import polygon

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

    def _round_clip_0_1(self, x):
        return x.round().clip(0, 1)

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
        X = []

        # Generate data
        for filepath in list_file_names:
            # Store normalized sample
            print(os.path.join(self.image_path, filepath))
            orig = Image.open(os.path.join(self.image_path, filepath))
            new = orig.resize(self.dim)
            new_rgb = new.convert('RGB')
            im = np.asarray(new_rgb, 'int')
            sample = self.preprocess_input(image=im)
            X.append(sample['image'])

        return np.array(X)

    def _generate_y(self, list_file_names):
        """Generates data containing batch_size masks
        :param list_file_names: list of image filenames to load
        :return: batch of masks
        """
        y = []

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
        for filename in list_file_names:
            # Store sample
            a = next(k for k in self.annotations if k['filename'] == filename)

            mask = np.zeros((*self.dim, self.n_classes), dtype=float)

            if 'regions' in a:
                polygons = [r for r in a['regions'].values()]

                for poly in polygons:
                    xp = [int(x * self.mask_scale_factor) for x in poly['shape_attributes']['all_points_x']]
                    yp = [int(y * self.mask_scale_factor) for y in poly['shape_attributes']['all_points_y']]
                    rr, cc = polygon(yp, xp)
                    class_id = self.classes.index(poly['region_attributes']['building'])
                    mask[rr, cc, class_id] = 1

            # add background if mask is not binary
            if mask.shape[-1] != 1:
                background = 1 - mask.sum(axis=-1, keepdims=True)
                mask = np.concatenate((mask, background), axis=-1)

            y.append(self._round_clip_0_1(mask))

        return np.array(y)


def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


model = sm.FPN('resnet101', classes=19, encoder_weights='imagenet', activation='softmax')

classes = [
    "flat",
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

model.load_weights('fpn_resnet101_weights.latest.h5')


def round_clip_0_1(x):
    return x.round().clip(0, 1)


train_gen = DataGenerator("deeproof-release/data/final-dataset/test", classes, 'resnet101', batch_size=1,
                          dim=(512, 512))

x, y = train_gen[0]

pr_mask_all = model.predict(x)
pr_mask = np.zeros((512, 512, 19))

for i in range(19):
    frame = round_clip_0_1(pr_mask_all[0, :, :, i])
    frame = frame * i
    pr_mask[:, :, i] = frame

pr_mask = np.sum(pr_mask, axis=-1)

actual_mask = np.zeros((512, 512, 19))

for i in range(19):
    frame = round_clip_0_1(y[0, :, :, i])
    frame = frame * i
    actual_mask[:, :, i] = frame

actual_mask = np.sum(actual_mask, axis=-1)

visualize(
    image=x[0, :, :, :],
    pr_mask=pr_mask,
    actual=actual_mask
)
#     # x = np.zeros((2, 512, 512, 3))
#     # x[0, :, :, :] = image
#     # x[1, :, :, :] = image
#     # pr_mask = model.predict(x)[0, :, :, 3]
#     #
#     # visualize(
#     #     image=denormalize(image.squeeze()),
#     #     #gt_mask=gt_mask.squeeze(),
#     #     pr_mask=pr_mask.squeeze(),
#     # )
