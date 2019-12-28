# Mask_RCNN (MRCNN) is a feature pyramid network (FPN) with a ResNet101 backbone
# Other models (Unet, FPN+Resnet-50) are trained in terrain_segmentation_others.py
import os
import time

import mrcnn.buildings
from mrcnn import model as modellib
from mrcnn import utils

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
model = modellib.MaskRCNN(mode="training", config=config, model_dir='models/')

print('Loading weights')
# Local path to trained weights file
COCO_MODEL_PATH = "mask_rcnn_coco.h5"
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# load the weights for COCO
model.load_weights(COCO_MODEL_PATH,
                   by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

print("Training network heads")
model.train(train_set, val_set,
            learning_rate=config.LEARNING_RATE,
            epochs=40,
            layers='heads')

# Training - Stage 2
# Finetune layers from ResNet stage 4 and up
print("Fine tune Resnet stage 4 and up")
model.train(train_set, val_set,
            learning_rate=config.LEARNING_RATE,
            epochs=100,
            layers='4+')

# Training - Stage 3
# Fine tune all layers
print("Fine tune all layers")
model.train(train_set, val_set,
            config.LEARNING_RATE / 10,
            epochs=160,
            layers='all')

print('Saving model')
filename = 'models/mask_rcnn_' + '.' + str(time.time())
model_path = filename + '.h5'
model.keras_model.save_weights(model_path)
