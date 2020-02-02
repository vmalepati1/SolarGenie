#!/usr/bin/python
# GoogleMapDownloader.py
# Created by Hayden Eskriett [http://eskriett.com]
#
# A script which when given a longitude, latitude and zoom level downloads a
# high resolution google map
# Find the associated blog post at: http://blog.eskriett.com/2013/07/19/downloading-google-maps/

import urllib.request
from PIL import Image
import os
import math
import mrcnn.buildings
from mrcnn import utils
import mrcnn.model as modellib
import numpy as np
from PIL import Image
from mrcnn.visualize import display_instances
import osmnx as ox
import segmentation_models as sm
from PIL import Image, ImageEnhance, ImageStat
import albumentations as A
import matplotlib.pyplot as plt

class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
        tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.
            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                url = 'http://mt0.google.com/vt/lyrs=s&hl=en&x=' + str(start_x + x) + '&y=' + str(start_y + y) + '&z=' + str(
                    self._zoom)

                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img

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
    lat, long = ox.geocode("3250 Twisted Branches Ln NE, Marietta, GA 30068")

    gmd = GoogleMapDownloader(lat, long, 19)

    print("The tile coorindates are {}".format(gmd.getXY()))

    try:
        # Get the high resolution image
        img = gmd.generateImage(tile_width=1, tile_height=1)

        preprocess_input = _get_preprocessing(sm.get_preprocessing('resnet101'))

        model = sm.FPN('resnet101', classes=19, encoder_weights='imagenet', activation='softmax')

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

        model.load_weights('fpn_resnet101_weights.latest.h5')

        x = []

        new = img.resize((512, 512))

        TOLERANCE = 5.0
        TARGET = 100

        updated = new

        if (brightness(new)) < TARGET:
            i = 1.0
            while True:
                updated = ImageEnhance.Brightness(new).enhance(i)
                print(brightness(updated))
                if abs(brightness(updated) - TARGET) < TOLERANCE:
                    break
                i += 0.05
        else:
            i = 1.0
            while True:
                updated = ImageEnhance.Brightness(new).enhance(i)
                print(brightness(updated))
                if abs(brightness(updated) - TARGET) < TOLERANCE:
                    break
                i -= 0.05

        # new = ImageEnhance.Brightness(new).enhance(1.1)
        updated = ImageEnhance.Contrast(updated).enhance(1.4)
        updated = ImageEnhance.Sharpness(updated).enhance(1.2)
        new_rgb = updated.convert('RGB')
        print(brightness(new_rgb))
        im = np.asarray(new_rgb, 'int')
        sample = preprocess_input(image=im)
        x.append(sample['image'])

        x = np.array(x)

        pr_mask_all = model.predict(x)
        pr_mask = np.zeros((512, 512, 19))

        for i in range(19):
            frame = round_clip_0_1(pr_mask_all[0, :, :, i])
            frame = frame * i
            pr_mask[:, :, i] = frame

        pr_mask = np.sum(pr_mask, axis=-1)

        visualize(
            image=x[0, :, :, :],
            pr_mask=pr_mask,
        )
    except IOError:
        print("Could not generate the image - try adjusting the zoom level and checking your coordinates")


if __name__ == '__main__':  main()