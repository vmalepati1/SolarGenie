import json
import os

import numpy as np
from sklearn.utils import class_weight

IMAGE_DIM = 512


def print_weights(dataset_dir, subset):
    # List of colors for mask
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
        "BG",
    ]

    # Train or validation dataset?
    assert subset in ["train", "val", "test"]
    dataset_dir = os.path.join(dataset_dir, subset)

    y_train = []

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

        for poly in polygons:
            y_train.append(poly['region_attributes']['building'])

        y_train.append('BG')

    class_weights = class_weight.compute_class_weight('balanced',
                                                      classes,
                                                      y_train)

    print(class_weights)

ROOF_DIR = "deeproof-release/data/final-dataset/"
print_weights(ROOF_DIR, 'train')
# print_weights(ROOF_DIR, 'val')
# print_weights(ROOF_DIR, 'test')
