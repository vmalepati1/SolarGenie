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
from solar_potential_analysis.solar_potential_analysis import get_pixelwise_solar_irradiance
from skimage.draw import polygon
import csv
import math
import parmap
import requests

# def process_pixelwise_solar_potential(point, gmd, origin_x, origin_y, optimal_map_scale, index, orientation, roof_pitch):
#
#     url = "http://developer.nrel.gov/api/solar/nsrdb_psm3_download.csv?api_key=fDFOk0vwiLzSJePAtLKg0oV8A8mex3yA9qYpdftK"
#
#     headers = {
#         'content-type': "application/x-www-form-urlencoded",
#         'cache-control': "no-cache"
#     }
#
#     x = point[0]
#     y = point[1]
#
#     lat, long = gmd.pixel_xy_to_lat_long(origin_x + x, origin_y + y, optimal_map_scale)
#
#     payload = "names=2018&leap_day=false&interval=30&utc=false&email=srinivasdeva3%40gmail.com&attributes=dhi%2Cdni%2Csolar_zenith_angle&wkt=POINT({}%20{})" \
#         .format(long, lat)
#
#     response = requests.request("POST", url, data=payload, headers=headers)
#
#     try:
#         data = response.text.splitlines()[index].split(',')
#     except IndexError:
#         return 0
#
#     beam_rad = float(data[6])
#     diff_rad = float(data[5])
#     solar_elevation_angle = math.radians(90 - float(data[7]))
#     solar_azimuth_angle = math.radians(float(data[7]))
#
#     return get_pixelwise_solar_irradiance(beam_rad, diff_rad, orientation, roof_pitch,
#                                                                solar_elevation_angle, solar_azimuth_angle)

if __name__ == '__main__':
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
    #
    # actual_mask = np.zeros((256, 256, 19))
    #
    # for i in range(18):
    #     frame = np.round(predicted_mask[0, :, :, i]).clip(0, 1)
    #     frame = frame * i
    #     actual_mask[:, :, i] = frame
    #
    # actual_mask = np.sum(actual_mask, axis=-1)
    #
    # plt.imshow(actual_mask)
    # plt.show()

    class_id_to_azimuth = {
        1: 0,
        2: 22.5,
        3: 45,
        4: 67.5,
        5: 90,
        6: 110.5,
        7: 135,
        8: 157.5,
        9: 180,
        10: 202.5,
        11: 225,
        12: 247.5,
        13: 270,
        14: 292.5,
        15: 315,
        16: 337.5,
    }

    SUMMER_SOLSTICE_INDEX = 6229

    building_outline = get_building_footprint(address, sq_m)

    solar_potential_map = np.zeros((mask_resolution, mask_resolution), dtype=np.float)

    for i, orientation in class_id_to_azimuth.items():

        contours, hierarchy = cv.findContours(predicted_mask[0, :, :, i], cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        tile_to_mask_scale = 256 / mask_resolution

        contours = list((map(np.squeeze, contours))) # removing redundant dimensions

        for j in range(0, len(contours)):
            contour = contours[j]

            try:
                points = np.round(contour * tile_to_mask_scale)
            except AssertionError:
                continue

            if points.ndim < 2:
                continue

            poly_points = []

            for point in points:
                x = point[0]
                y = point[1]

                lat, long = gmd.pixel_xy_to_lat_long(origin_x + x, origin_y + y, optimal_map_scale)
                poly_points.append((long, lat))

            pot_rooftop_planar = geometry.Polygon(poly_points)

            if pot_rooftop_planar.intersects(building_outline):
                rr, cc = polygon([point[1] for point in contour], [point[0] for point in contour])

                # solar_potential_map[rr, cc] = \
                #     parmap.map(process_pixelwise_solar_potential, zip(rr, cc), gmd, origin_x, origin_y, optimal_map_scale, SUMMER_SOLSTICE_INDEX, orientation, roof_pitch)


    plt.subplot(1, 2, 1)
    plt.imshow(gmd.generateImage(tile_width=1, tile_height=1))
    plt.subplot(1, 2, 2)
    plt.imshow(solar_potential_map, cmap="Wistia")
    plt.show()

    # x = [[point[0][0], point[0][1]] for point in contours[2]]
    # print(x)
    #
    # poly = geometry.Polygon(x)
    #
    # plt.plot(*poly.exterior.xy)
    # plt.show()