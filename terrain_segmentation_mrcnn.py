import time

import mrcnn.buildings
from mrcnn import model as modellib

config = mrcnn.buildings.BuildingConfig()
ROOF_DIR = "deeproof-release/data/final-dataset/"

# Load datasets
# Get the datasets from the releases page
train_set = mrcnn.buildings.BuildingDataset()
train_set.load_building(ROOF_DIR, "train")
train_set.prepare()

val_set = mrcnn.buildings.BuildingDataset()
val_set.load_building(ROOF_DIR, "val")
val_set.prepare()

print("Loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir='./')

print('Loading weights')
# load the weights for COCO
model.load_weights('mask_rcnn_coco.h5',
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

print('Training model')

model.train(train_set, val_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
history = model.keras_model.history.history

print('Saving model')
filename = 'models/mask_rcnn_' + '.' + str(time.time())
model_path = filename + '.h5'
model.keras_model.save_weights(model_path)
