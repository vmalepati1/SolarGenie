import mrcnn.buildings
from mrcnn import utils
import mrcnn.model as modellib
import numpy as np
import skimage.io
from mrcnn.visualize import display_instances
import cv2

config = mrcnn.buildings.BuildingConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)

# load coco model weights
model.load_weights('MRCNN_weights.best_ST4.h5', by_name=True)

img = skimage.io.imread('deeproof-release/data/final-dataset/test/522749666.270.jpg')

# make prediction
results = model.detect([img, img], verbose=1)

class_names = [
    "BG", "flat", "dome", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "tree"
]

# get dictionary for first prediction
r = results[0]
print(r['class_ids'])
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])