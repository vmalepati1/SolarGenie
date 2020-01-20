import osmnx as ox
from shapely.geometry import Point

import numpy as np
import random
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models as sm

class TileServer(object):

    MIN_LATITUDE = -85.05112878
    MAX_LATITUDE = 85.05112878
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180

    def __init__(self):
        self.imdict = {}
        self.surfdict = {}
        self.layers = 'SATELLITE'
        self.path = './'
        self.urltemplate = 'http://ecn.t{4}.tiles.virtualearth.net/tiles/{3}{5}?g=0'
        self.layerdict = {'SATELLITE': 'a', 'HYBRID': 'h', 'ROADMAP': 'r'}

    def tiletoquadkey(self, xi, yi, z):
        quadKey = ''
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << (i - 1)
            if(xi & mask) != 0:
                digit += 1
            if(yi & mask) != 0:
                digit += 2
            quadKey += str(digit)
        return quadKey

    def loadimage(self, response, tilekey):
        im = Image.open(BytesIO(response.content))
        self.imdict[tilekey] = im
        return self.imdict[tilekey]

    def tile_as_image(self, xi, yi, zoom):
        tilekey = (xi, yi, zoom)

        try:
            result = self.imdict[tilekey]
        except:
            server = random.choice(range(1,4))
            quadkey = self.tiletoquadkey(*tilekey)
            url = self.urltemplate.format(xi, yi, zoom, self.layerdict[self.layers], server, quadkey)
            response = requests.get(url)
            result = self.loadimage(response, tilekey)

        return result

    def map_size(self, detail):
        return 256 << detail

    def lat_long_to_pixel_xy(self, latitude, longitude, detail):
        latitude = np.clip(latitude, self.MIN_LATITUDE, self.MAX_LATITUDE)
        longitude = np.clip(longitude, self.MIN_LONGITUDE, self.MAX_LONGITUDE)

        x = (longitude + 180) / 360
        sinLatitude = np.sin(latitude * np.pi / 180)
        y = 0.5 - np.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * np.pi)

        mapSize = self.map_size(detail)
        pixelX = np.clip(x * mapSize + 0.5, 0, mapSize - 1).astype(int)
        pixelY = np.clip(y * mapSize + 0.5, 0, mapSize - 1).astype(int)

        return pixelX, pixelY

    def pixel_xy_to_lat_long(self, pixel_x, pixel_y, detail):
        mapSize = self.map_size(detail)
        x = (np.clip(pixel_x, 0, mapSize - 1) / mapSize) - 0.5
        y = 0.5 - (np.clip(pixel_y, 0, mapSize - 1) / mapSize)

        latitude = 90 - 360 * np.arctan(np.exp(-y * 2 * np.pi)) / np.pi
        longitude = 360 * x
        return latitude, longitude

    def pixel_xy_to_tile_xy(self, pixel_x, pixel_y):
        tileX = pixel_x // 256
        tileY = pixel_y // 256
        return tileX, tileY

    def tile_xy_to_pixel_xy(self, tile_x, tile_y):
        pixelX = tile_x * 256
        pixelY = tile_y * 256
        return pixelX, pixelY

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

if __name__ == "__main__":
    ts = TileServer()

    # s = 16
    #
    # lat, long = ox.geocode("3250 Twisted Branches Ln NE, Marietta, GA 30068")
    # scale = np.log2(s * np.cos(lat * np.pi/180) * 2 * np.pi * 6378137)
    #
    # print(scale)

    scale = 19
    lat, long = ox.geocode("3250 Twisted Branches Ln NE, Marietta, GA 30068")
    print(lat)
    print(long)
    x, y = ts.lat_long_to_pixel_xy(lat, long, scale)
    tile_x, tile_y = ts.pixel_xy_to_tile_xy(x, y)
    img = ts.tile_as_image(tile_x, tile_y, scale)
    img = img.resize((512, 512))
    img = np.array(img)

    model = sm.FPN('resnet101', classes=20, encoder_weights='imagenet', activation='softmax')

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

    model.load_weights('fpn_resnet101_weights.best_e91.h5')


    def round_clip_0_1(x):
        return np.ceil(x).clip(0, 1).astype('float')

    x = np.zeros((1, 512, 512, 3))
    x[0, :, :, :] = img

    pr_mask = model.predict(x)
    visualize(
        image=img.squeeze(),
        pr_mask=round_clip_0_1(1 - pr_mask.sum(axis=-1, keepdims=True)[0, :, :, 0]),
    )