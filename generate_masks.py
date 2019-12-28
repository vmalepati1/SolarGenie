import json
import os

from PIL import Image, ImageDraw
from matplotlib import image

IMAGE_DIM = 512


def gen_masks(dataset_dir, subset):
    # List of colors for mask
    class_colormap = {
        "N": "#eb3434",
        "NNE": "#eb8f34",
        "NE": "#ebb134",
        "ENE": "#ebd634",
        "E": "#d6eb34",
        "ESE": "#abeb34",
        "SE": "#34eb83",
        "SSE": "#34ebdf",
        "S": "#34d0eb",
        "SSW": "#349feb",
        "SW": "#3471eb",
        "WSW": "#344feb",
        "W": "#a534eb",
        "WNW": "#d634eb",
        "NW": "#eb34d0",
        "NNW": "#eb349f",
        "tree": "#eb3489",
        "flat": "#eb3468",
        "dome": "#eb344f",
    }

    # Train or validation dataset?
    assert subset in ["train", "val", "test"]
    dataset_dir = os.path.join(dataset_dir, subset)

    os.mkdir(dataset_dir + '/masks')

    # Load annotations
    # VGG Image Annotator saves each image in the form:
    # { 'filename': '28503151_5b5b7ec140_b.jpg',
    #   'regions': {
    #       '0': {
    #           'region_attributes': {},
    #           'shape_attributes': {
    #               'all_points_x': [...],
    #               'all_points_y': [...],
    #               'name': 'polygon'}},
    #       ... more regions ...
    #   },
    #   'size': 100202
    # }
    # We mostly care about the x and y coordinates of each region
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())  # don't need the dict keys

    # The VIA tool saves images in the JSON even if they don't have any
    # annotations. Skip unannotated images.
    annotations = [a for a in annotations if a['regions']]

    # Add images
    for a in annotations:
        # Get the x, y coordinaets of points of the polygons that make up
        # the outline of each object instance. There are stores in the
        # shape_attributes (see json format above)
        polygons = [r for r in a['regions'].values()]
        # load_mask() needs the image size to convert polygons to masks.
        # Unfortunately, VIA doesn't include it in JSON, so we must read
        # the image. This is only managable since the dataset is tiny.
        image_path = os.path.join(dataset_dir, a['filename'])
        read_image = image.imread(image_path)
        height, width = read_image.shape[:2]

        if width != IMAGE_DIM or height != IMAGE_DIM:
            raise

        mask = Image.new('RGB', (width, height))
        draw_canvas = ImageDraw.Draw(mask)

        for poly in polygons:
            xy = tuple(zip(poly['shape_attributes']['all_points_x'], poly['shape_attributes']['all_points_y']))
            draw_canvas.polygon(xy, fill=class_colormap[poly['region_attributes']['building']])

        mask.save(dataset_dir + '/masks/mask_' + a['filename'])


ROOF_DIR = "deeproof-release/data/final-dataset/"
# gen_masks(ROOF_DIR, 'train')
# gen_masks(ROOF_DIR, 'val')
gen_masks(ROOF_DIR, 'test')
