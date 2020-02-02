import albumentations as A
import keras
import numpy as np
import segmentation_models as sm
from PIL import ImageEnhance, ImageStat
from data_generator import DataGenerator
from orientation_error_metric import mean_orientation_error
from google_map_downloader import GoogleMapDownloader
import matplotlib.pyplot as plt
from geopandas.tools import geocode



class TerrainSegmentation:

    def __init__(self, classes, class_id_to_azimuth=None, optim=None, loss=None,
                 model_path='../models/fpn_resnet101_weights.latest.h5'):
        self.classes = classes
        self.n_classes = len(self.classes) + 1
        self.model = sm.FPN('resnet101', classes=self.n_classes, encoder_weights='imagenet', activation='softmax')
        self.model.load_weights(model_path)
        self.preprocess_input = self._get_preprocessing(sm.get_preprocessing('resnet101'))
        self.class_id_to_azimuth = class_id_to_azimuth
        self.optim = optim
        self.loss = loss

    def evaluate(self):
        if self.optim is None or self.loss is None or self.class_id_to_azimuth is None:
            raise Exception('Must pass an azimuth dict, optimizer and loss to evaluate model!')

        test_gen = DataGenerator('../deeproof-release/data/final-dataset/test', self.classes, 'resnet101', batch_size=1,
                                 dim=(512, 512), mask_scale_factor=1.0)
        self.model.compile(self.optim, self.loss,
                           metrics=[mean_orientation_error(self.class_id_to_azimuth, batch_size=1)])
        print(self.model.evaluate_generator(test_gen, steps=10))

    def predict_mask(self, satellite_img, target_brightness=85):
        new_rgb = satellite_img.convert('RGB')
        new_rgb = ImageEnhance.Brightness(new_rgb).enhance(target_brightness / self.brightness(new_rgb))
        new_rgb = ImageEnhance.Contrast(new_rgb).enhance(1.4)
        new_rgb = ImageEnhance.Sharpness(new_rgb).enhance(1.2)
        im = np.asarray(new_rgb, 'int')
        sample = self.preprocess_input(image=im)

        return self.round_clip_0_1(self.model.predict(np.array([sample['image']])))

    def round_clip_0_1(self, x):
        return x.round().clip(0, 1)

    def _get_preprocessing(self, preprocessing_fn):
        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)

    def brightness(self, im):
        stat = ImageStat.Stat(im)
        return stat.mean[0]


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

g = geocode(["901 Highbury Ln NE, Marietta, GA 30068"], timeout=5.0)
lat = g.geometry[0].y
long = g.geometry[0].x

gmd = GoogleMapDownloader(lat, long, 19)

# Get the high resolution image
img = gmd.generateImage(tile_width=1, tile_height=1)

ts = TerrainSegmentation(classes)
plt.imshow(ts.predict_mask(img)[0, :, :, 18])
plt.show()