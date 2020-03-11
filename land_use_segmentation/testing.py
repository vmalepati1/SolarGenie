import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from solar_potential_analysis.solar_potential_analysis import get_pixelwise_solar_irradiance
from skimage.draw import polygon
import pvlib
from topology_estimation.get_elevation import elevation_function
import pandas as pd
import math
import parmap
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import normalize

def process_pixelwise_solar_potential(point, orientation, roof_pitch, times, altitude):
    x = point[1]
    y = point[0]

    latitude, longitude = pixel_xy_to_lat_long(x, y)

    solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
    pressure = pvlib.atmosphere.alt2pres(altitude)
    am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
    tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
    cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
                                 dni_extra=dni_extra, altitude=altitude)

    beam_rad = cs['dni']
    diff_rad = cs['dhi']
    solar_elevation_angle = math.radians(90 - solpos['apparent_zenith'])
    solar_azimuth_angle = math.radians(solpos['azimuth'])

    return get_pixelwise_solar_irradiance(beam_rad, diff_rad, orientation, roof_pitch,
                                                               solar_elevation_angle, solar_azimuth_angle)

def pixel_xy_to_lat_long(pixel_x, pixel_y):
    upper_left_corner = (37.060708, -78.643969)
    bottom_right_corner = (37.051190, -78.635577)

    image_width = 520
    image_height = 516

    upper_lat = upper_left_corner[0]
    lower_lat = bottom_right_corner[0]
    lat_range = upper_lat - lower_lat

    left_long = upper_left_corner[1]
    right_long = bottom_right_corner[1]
    long_range = right_long - left_long

    displacement_lat = (pixel_y / image_height) * lat_range
    displacement_long = (pixel_x / image_width) * long_range

    return upper_lat - displacement_lat, left_long + displacement_long

if __name__ == '__main__':

    img = cv2.imread('images/segmentation_result.PNG')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    low = np.array([100, 200, 100])
    high = np.array([188, 255, 188])

    mask = cv2.inRange(img, low, high)

    solar_potential_map = np.zeros((516, 520), dtype=np.float)

    times = pd.date_range('06/21/2018 14:00', periods=1, freq='20min')
    times = times.tz_localize('EST')

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    contours = list((map(np.squeeze, contours))) # removing redundant dimensions

    lat = 37.055522
    long = -78.638640
    threshold_area = 400
    altitude = elevation_function((lat, long))

    for j in range(0, len(contours)):
        contour = contours[j]

        try:
            area = cv2.contourArea(contour)
        except cv2.error as e:
            continue

        if area < threshold_area:
            continue

        try:
            points = np.round(contour)
        except AssertionError:
            continue

        if points.ndim < 2:
            continue

        print('Contour ' + str(j) + ' of ' + str(len(contours)))

        rr, cc = polygon([point[1] for point in contour], [point[0] for point in contour])

        parallel_results = np.asarray(
            parmap.map(process_pixelwise_solar_potential, zip(rr, cc), math.pi, 0, times, altitude)).squeeze()

        solar_potential_map[rr, cc] = parallel_results

    masked_data = np.ma.masked_where(solar_potential_map == 0, solar_potential_map)

    image = Image.open('images/original.png')

    plt.imshow(np.array(image))
    plt.imshow(masked_data, cmap="autumn")
    plt.show()