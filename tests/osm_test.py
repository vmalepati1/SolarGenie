import random
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import requests
import segmentation_models as sm
from PIL import Image, ImageEnhance, ImageStat
import albumentations as A
from geopandas.tools import geocode


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

if __name__ == "__main__":
    ts = TileServer()

    s = 16

    lat, long = ox.geocode("3473 Princeton Corners Dr, Marietta, GA 30062")
    scale = np.log2(s * np.cos(lat * np.pi/180) * 2 * np.pi * 6378137)

    print(scale)

    scale = 19
    print(lat)
    print(long)
    x, y = ts.lat_long_to_pixel_xy(lat, long, scale)
    tile_x, tile_y = ts.pixel_xy_to_tile_xy(x, y)
    orig = ts.tile_as_image(tile_x, tile_y, scale)
    # img = img.resize((512, 512))
    # img = ImageEnhance.Brightness(img).enhance(1.25)
    # img = ImageEnhance.Contrast(img).enhance(1.4)
    # # img = ImageEnhance.Sharpness(img).enhance(1.2)
    # orig = np.array(img)
    #
    # img = np.array(img)
    #

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

    new = orig.resize((512, 512))

    TOLERANCE = 5.0

    updated = new
    i = 1.0
    while True:
        updated = ImageEnhance.Brightness(new).enhance(i)
        print(brightness(updated))
        if abs(brightness(updated) - 111) < TOLERANCE:
            break
        i += 0.1

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

    # fig, ax = plt.subplots()
    # plt.subplots_adjust(left=0.25, bottom=0.25)
    # l = plt.imshow(pr_mask)
    # ax.margins(x=0)
    #
    # from matplotlib.widgets import Slider, Button, RadioButtons
    #
    # axcolor = 'lightgoldenrodyellow'
    # axbri = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    # axcon = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    # axsha = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    #
    # sbri = Slider(axbri, 'brightness', 0.0, 5.0, valinit=1.1, valstep=0.1)
    # scon = Slider(axcon, 'contrast', 0.0, 5.0, valinit=1.4, valstep=0.1)
    # ssha = Slider(axsha, 'sharpness', 0.0, 5.0, valinit=1.2, valstep=0.1)
    #
    #
    # def update(val):
    #     x = []
    #     new = orig.resize((512, 512))
    #     new = ImageEnhance.Brightness(new).enhance(sbri.val)
    #     new = ImageEnhance.Contrast(new).enhance(scon.val)
    #     new = ImageEnhance.Sharpness(new).enhance(ssha.val)
    #     new_rgb = new.convert('RGB')
    #     im = np.asarray(new_rgb, 'int')
    #     sample = preprocess_input(image=im)
    #     x.append(sample['image'])
    #
    #     x = np.array(x)
    #
    #     pr_mask_all = model.predict(x)
    #     pr_mask = np.zeros((512, 512, 19))
    #
    #     for i in range(19):
    #         frame = round_clip_0_1(pr_mask_all[0, :, :, i])
    #         frame = frame * i
    #         pr_mask[:, :, i] = frame
    #
    #     pr_mask = np.sum(pr_mask, axis=-1)
    #
    #     l.set_data(pr_mask)
    #     fig.canvas.draw_idle()
    #
    #
    # sbri.on_changed(update)
    # scon.on_changed(update)
    # ssha.on_changed(update)
    #
    # plt.show()