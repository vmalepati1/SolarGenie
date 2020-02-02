# from topology_estimation.get_building_data import get_building_data
# from topology_estimation.find_map_scale import find_map_scale_from_area
# from geopandas.tools import geocode
# from terrain_segmentation.terrain_segmentation import TerrainSegmentation
# from terrain_segmentation.google_map_downloader import GoogleMapDownloader

# address = '901 Highbury Ln, Marietta, GA'
# zipcode = '30068'
#
# g = geocode([address], timeout=5.0)
# lat = g.geometry[0].y
# long = g.geometry[0].x
#
# sq_m, roof_pitch = get_building_data(address, zipcode)
#
# map_scale_factor_of_safety = 10.0
#
# optimal_map_scale = find_map_scale_from_area(lat, sq_m * map_scale_factor_of_safety)
#
# gmd = GoogleMapDownloader(lat, long, optimal_map_scale)
#
# # Get the high resolution image
# img = gmd.generateImage(tile_width=1, tile_height=1)
#
# classes = [
#     "flat",
#     "N",
#     "NNE",
#     "NE",
#     "ENE",
#     "E",
#     "ESE",
#     "SE",
#     "SSE",
#     "S",
#     "SSW",
#     "SW",
#     "WSW",
#     "W",
#     "WNW",
#     "NW",
#     "NNW",
#     "tree",
# ]
#
# ts = TerrainSegmentation(classes, model_path='models/fpn_resnet101_weights.latest.h5')
#
# np.savez('saved_masks/first_test.npz', mask=ts.predict_mask(img))

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from shapely import geometry
from geopandas.tools import geocode
from terrain_segmentation.google_map_downloader import GoogleMapDownloader
from topology_estimation.extract_geometry import get_building_footprint
from topology_estimation.get_building_data import get_building_data

optimal_map_scale = 19

address = '901 Highbury Ln, Marietta, GA'
zipcode = '30068'

g = geocode([address], timeout=5.0)
lat = g.geometry[0].y
long = g.geometry[0].x

sq_m, roof_pitch = get_building_data(address, zipcode)

gmd = GoogleMapDownloader(lat, long, optimal_map_scale)
tile_x, tile_y = gmd.getXY()
origin_x, origin_y = gmd.tile_xy_to_pixel_xy(tile_x, tile_y)

predicted_mask = np.load('saved_masks/first_test.npz')['mask']
predicted_mask = predicted_mask.astype(np.uint8)
mask_resolution = predicted_mask.shape[1]

contours, hierarchy = cv.findContours(predicted_mask[0, :, :, 18], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

tile_to_mask_scale = 256 / mask_resolution

building_outline = get_building_footprint(address, sq_m)

print(building_outline)

contours = list((map(np.squeeze, contours))) # removing redundant dimensions

for i in range(0, len(contours)):
    try:
        points = np.round(contours[i] * tile_to_mask_scale)
    except AssertionError:
        continue

    if points.ndim < 2:
        continue

    poly = geometry.Polygon(points)

    plt.plot(*poly.exterior.xy)

    poly_points = []

    for point in points:
        x = point[0]
        y = point[1]

        lat, long = gmd.pixel_xy_to_lat_long(origin_x + x, origin_y + y, optimal_map_scale)
        poly_points.append((long, lat))

    pot_rooftop_planar = geometry.Polygon(poly_points)

    if pot_rooftop_planar.intersects(building_outline):
        print('Intersection at {}'.format(i))

plt.show()

# x = [[point[0][0], point[0][1]] for point in contours[2]]
# print(x)
#
# poly = geometry.Polygon(x)
#
# plt.plot(*poly.exterior.xy)
# plt.show()