import numpy as np
from PIL import Image

import mrcnn.buildings
import mrcnn.model as modellib
from mrcnn.visualize import display_instances

config = mrcnn.buildings.BuildingConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir='./', config=config)

# load coco model weights
model.load_weights('MRCNN_weights.best_ST4.h5', by_name=True)

img = Image.open('test.png').convert('RGB')
img = img.resize((512, 512))
img = np.array(img)

# make prediction
results = model.detect([img, img], verbose=1)

class_names = [
    "BG", "flat", "dome", "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW",
    "NNW", "tree"
]

# get dictionary for first prediction
r = results[0]
print(r['class_ids'])
# show photo with bounding boxes, masks, class labels and scores
display_instances(img, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
