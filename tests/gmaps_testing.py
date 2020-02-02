#!/usr/bin/python
# GoogleMapDownloader.py
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import math
import os
import urllib.request

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models as sm
from PIL import Image, ImageEnhance, ImageStat
from geopandas.tools import geocode
from terrain_segmentation.google_map_downloader import GoogleMapDownloader


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


def round_clip_0_1(x):
    return x.round().clip(0, 1)


def _get_preprocessing(preprocessing_fn):
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


def brightness(im):
    stat = ImageStat.Stat(im)
    return stat.mean[0]


def main():
    # Create a new instance of GoogleMap Downloader
    # Geocoding not accurate as of now
    # lat, long = ox.geocode("3259 Twisted Branches Ln NE, Marietta, GA 30068")

    TARGET_BRIGHTNESS = 70

    g = geocode(["901 Highbury Ln NE, Marietta, GA 30068"], timeout=5.0)
    lat = g.geometry[0].y
    long = g.geometry[0].x

    gmd = GoogleMapDownloader(lat, long, 19)

    print("The tile coordinates are {}".format(gmd.getXY()))

    # Get the high resolution image
    img = gmd.generateImage(tile_width=1, tile_height=1)

    preprocess_input = _get_preprocessing(sm.get_preprocessing('resnet101'))

    model = sm.FPN('resnet101', classes=19, encoder_weights='imagenet', activation='softmax')

    model.load_weights('models/fpn_resnet101_weights.latest.h5')

    x = []

    img = img.resize((512, 512))
    new_rgb = img.convert('RGB')
    new_rgb = ImageEnhance.Brightness(new_rgb).enhance(TARGET_BRIGHTNESS / brightness(new_rgb))
    new_rgb = ImageEnhance.Contrast(new_rgb).enhance(1.4)
    new_rgb = ImageEnhance.Sharpness(new_rgb).enhance(1.2)
    im = np.asarray(new_rgb, 'int')
    sample = preprocess_input(image=im)
    x.append(sample['image'])

    x = np.array(x)

    pr_mask_all = model.predict(x)
    pr_mask = np.zeros((512, 512, 19))

    for i in range(19):
        frame = np.where(pr_mask_all[0, :, :, i] > 0.1, 1, 0).clip(0, 1)
        frame = frame * i
        pr_mask[:, :, i] = frame

    pr_mask = np.sum(pr_mask, axis=-1)

    visualize(
        image=x[0, :, :, :],
        pr_mask=pr_mask,
    )


if __name__ == '__main__':  main()
