import json

import segmentation_models as sm
import keras
from time import time
from PIL import Image, ImageDraw
from matplotlib import image
import os
import numpy as np

# Number of classes (including background)
NUM_CLASSES = 1 + 16 + 2 + 1
IMAGE_DIM = 512

models = {
    # name, model instance
    'unet_resnet50': sm.Unet('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'unet_resnet101': sm.Unet('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet50': sm.FPN('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet101': sm.FPN('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
}

def gen_images_and_masks(dataset_dir, subset):
    images = []
    masks = []

    # List of colors for mask
    class_colormap = {
        "N": "#eb3434",
        "NNE": "#eb8f34",
        "NE": "#ebb134",
        "ENE": "#ebd634",
        "E": "#d6eb34",
        "ESE": "#abeb34",
        "SE": "#34eb83",
        "SSE": "#34ebdf",
        "S": "#34d0eb",
        "SSW": "#349feb",
        "SW": "#3471eb",
        "WSW": "#344feb",
        "W": "#a534eb",
        "WNW": "#d634eb",
        "NW": "#eb34d0",
        "NNW": "#eb349f",
        "tree": "#eb3489",
        "flat": "#eb3468",
        "dome": "#eb344f",
    }

    # Train or validation dataset?
    assert subset in ["train", "val", "test"]
    dataset_dir = os.path.join(dataset_dir, subset)

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
    # We mostly care about the x and y coordinates of each region
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. There are stores in the
        # shape_attributes (see json format above)
        polygons = [r for r in a['regions'].values()]
        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.
        image_path = os.path.join(dataset_dir, a['filename'])
        read_image = image.imread(image_path)
        images.append(read_image)
        height, width = read_image.shape[:2]

        if width != IMAGE_DIM or height != IMAGE_DIM:
            raise

        mask = Image.new('RGB', (width, height))
        draw_canvas = ImageDraw.Draw(mask)

        for poly in polygons:
            xy = tuple(zip(poly['shape_attributes']['all_points_x'], poly['shape_attributes']['all_points_y']))
            draw_canvas.polygon(xy, fill=class_colormap[poly['region_attributes']['building']])

        masks.append(np.array(mask))

    return images, masks

# Run to generate numpy dataset
# ROOF_DIR = "deeproof-release/data/final-dataset/"
# X_train, y_train = gen_images_and_masks(ROOF_DIR, 'train')
# X_val, y_val = gen_images_and_masks(ROOF_DIR, 'val')
#
# np.savez('deeproof-release/data/final-dataset/building_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

for name, model in models.items():
    print('Compiling ' + name)
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    filepath = "models/" + name + ".best.hdf5"

    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir="models/logs/{}".format(time()),
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True),
    ]

    print('Loading data')
    dataset = np.load('deeproof-release/data/final-dataset/building_data.npz')
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']

    print('Training ' + name)
    model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,
        epochs=100,
        validation_data=(X_val, y_val),
    )

