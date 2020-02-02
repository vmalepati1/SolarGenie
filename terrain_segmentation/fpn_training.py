import os
from time import time

import keras
import segmentation_models as sm
from terrain_segmentation.data_generator import DataGenerator
import os
from time import time

import keras
import segmentation_models as sm

from terrain_segmentation.data_generator import DataGenerator

# Number of classes (including background)

# figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

TRAINING_IMAGE_DIR = 'deeproof-release/data/final-dataset/train'
VAL_IMAGE_DIR = 'deeproof-release/data/final-dataset/val'

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


def get_num_files_in_dir(dir):
    return len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])


NUMBER_OF_TRAINING_IMAGES = get_num_files_in_dir(TRAINING_IMAGE_DIR) - 1
NUMBER_OF_VAL_IMAGES = get_num_files_in_dir(VAL_IMAGE_DIR) - 1
print(NUMBER_OF_TRAINING_IMAGES)
print(NUMBER_OF_VAL_IMAGES)
IMAGE_DIM = 512
BATCH_SIZE = 2
LR = 0.001


def lr_decay(epoch):
    if epoch < 200:
        return LR
    elif epoch >= 200 and epoch < 320:
        return LR / 10
    else:
        return LR / 100


# Ideal calculations
# STEPS_PER_EPOCH = NUMBER_OF_TRAINING_IMAGES // BATCH_SIZE
# VAL_STEPS = NUMBER_OF_VAL_IMAGES // BATCH_SIZE

# "Slow-cook" dataset
STEPS_PER_EPOCH = 500
VAL_STEPS = 50

n_classes = 1 if len(classes) == 1 else (len(classes) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

print(n_classes)

backbone = 'resnet101'
name = 'fpn_' + backbone
model = sm.FPN(backbone, classes=n_classes, activation=activation, encoder_weights='imagenet',
               pyramid_aggregation='concat')

# define optomizer
optim = keras.optimizers.SGD(LR, momentum=0.9)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor
# set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
class_weights = [0.41432617, 1.08753001, 1.09721607, 3.24666718, 1.13021085, 1.08753001,
                 1.09721607, 3.24666718, 1.13021085, 1.08753001, 1.09721607, 3.24666718,
                 1.13021085, 1.08753001, 1.09721607, 3.24666718, 1.13021085, 0.32764091,
                 0.69469727]

dice_loss = sm.losses.DiceLoss(class_weights=class_weights)
ce_loss = sm.losses.CategoricalCELoss(class_weights=class_weights)
total_loss = dice_loss + ce_loss

# actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
# total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)

# Callbacks
callbacks = [
    keras.callbacks.TensorBoard(log_dir="models/others_logs/{}".format(time()),
                                histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint('models/' + name + '_weights.latest.h5',
                                    verbose=1, save_weights_only=True),
    keras.callbacks.ModelCheckpoint('models/' + name + '_weights.best.h5',
                                    verbose=1, save_best_only=True, save_weights_only=True, mode='min'),
    keras.callbacks.LearningRateScheduler(lr_decay)
]

print('Creating generators')

training_generator = DataGenerator(TRAINING_IMAGE_DIR, classes, backbone, batch_size=BATCH_SIZE,
                                   dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM / 512)
validation_generator = DataGenerator(VAL_IMAGE_DIR, classes, backbone, batch_size=BATCH_SIZE,
                                     dim=(IMAGE_DIM, IMAGE_DIM), mask_scale_factor=IMAGE_DIM / 512)

print('Training ' + name)
model.fit_generator(
    training_generator,
    verbose=1,
    epochs=480,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    validation_data=validation_generator,
    validation_steps=VAL_STEPS,
)
