import segmentation_models as sm
import keras
from time import time
from PIL import Image, ImageDraw

# Number of classes (including background)
NUM_CLASSES = 1 + 16 + 2 + 1

models = {
    # name, model instance
    'unet_resnet50': sm.Unet('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'unet_resnet101': sm.Unet('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet50': sm.FPN('resnet50', classes=NUM_CLASSES, encoder_weights='imagenet'),
    'fpn_resnet101': sm.FPN('resnet101', classes=NUM_CLASSES, encoder_weights='imagenet'),
}

def gen_images_and_masks(via_directory):
    class_colormap = {
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
        "tree"
    }

for name, model in models.items():
    print('Compiling ' + name)
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )

    print('Creating generators')
    # Data generators
    train_generator = modellib.data_generator(train_set, config, shuffle=True,
                                     batch_size=config.BATCH_SIZE)
    val_generator = modellib.data_generator(val_set, config, shuffle=True,
                                   batch_size=config.BATCH_SIZE)

    filepath = "models/" + name + ".best.hdf5"

    # Callbacks
    callbacks = [
        keras.callbacks.TensorBoard(log_dir="models/logs/{}".format(time()),
                                    histogram_freq=0, write_graph=True, write_images=False),
        keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True),
    ]

    print('Training ' + name)
    model.fit_generator(
        train_generator,
        epochs=100,
        steps_per_epoch=config.STEPS_PER_EPOCH,
        callbacks=callbacks,
        validation_data=val_generator,
        validation_steps=config.VALIDATION_STEPS,
        max_queue_size=100,
        workers=0,
        use_multiprocessing=False,
    )

