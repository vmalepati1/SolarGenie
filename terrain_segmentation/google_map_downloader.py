from PIL import Image
import math
import urllib
import os
import numpy as np

class GoogleMapDownloader:
    """
        A class which generates high resolution google maps images given
        a longitude, latitude and zoom level
    """

    def __init__(self, lat, lng, zoom=12):
        """
            GoogleMapDownloader Constructor
            Args:
                lat:    The latitude of the location required
                lng:    The longitude of the location required
                zoom:   The zoom level of the location required, ranges from 0 - 23
                        defaults to 12
        """
        self._lat = lat
        self._lng = lng
        self._zoom = zoom

    def getXY(self):
        """
            Generates an X,Y tile coordinate based on the latitude, longitude
            and zoom level
            Returns:    An X,Y tile coordinate
        """

        tile_size = 256

        # Use a left shift to get the power of 2
        # i.e. a zoom level of 2 will have 2^2 = 4 tiles
        numTiles = 1 << self._zoom

        # Find the x_point given the longitude
        point_x = (tile_size / 2 + self._lng * tile_size / 360.0) * numTiles // tile_size

        # Convert the latitude to radians and take the sine
        sin_y = math.sin(self._lat * (math.pi / 180.0))

        # Calulate the y coorindate
        point_y = ((tile_size / 2) + 0.5 * math.log((1 + sin_y) / (1 - sin_y)) * -(
                tile_size / (2 * math.pi))) * numTiles // tile_size

        return int(point_x), int(point_y)

    def map_size(self, detail):
        return 256 << detail

    def lat_long_to_pixel_xy(self, latitude, longitude, detail):
        latitude = np.clip(latitude, self.MIN_LATITUDE, self.MAX_LATITUDE)
        longitude = np.clip(longitude, self.MIN_LONGITUDE, self.MAX_LONGITUDE)

        x = (longitude + 180) / 360
        sinLatitude = np.sin(latitude * np.pi / 180)
        y = 0.5 - np.log((1 + sinLatitude) / (1 - sinLatitude)) / (4 * np.pi)

        mapSize = self.map_size(detail)
        pixelX = np.clip(x * mapSize + 0.5, 0, mapSize - 1).astype(int)
        pixelY = np.clip(y * mapSize + 0.5, 0, mapSize - 1).astype(int)

        return pixelX, pixelY

    def pixel_xy_to_lat_long(self, pixel_x, pixel_y, detail):
        mapSize = self.map_size(detail)
        x = (np.clip(pixel_x, 0, mapSize - 1) / mapSize) - 0.5
        y = 0.5 - (np.clip(pixel_y, 0, mapSize - 1) / mapSize)

        latitude = 90 - 360 * np.arctan(np.exp(-y * 2 * np.pi)) / np.pi
        longitude = 360 * x
        return latitude, longitude

    def pixel_xy_to_tile_xy(self, pixel_x, pixel_y):
        tileX = pixel_x // 256
        tileY = pixel_y // 256
        return tileX, tileY

    def tile_xy_to_pixel_xy(self, tile_x, tile_y):
        pixelX = tile_x * 256
        pixelY = tile_y * 256
        return pixelX, pixelY

    def generateImage(self, **kwargs):
        """
            Generates an image by stitching a number of google map tiles together.
            Args:
                start_x:        The top-left x-tile coordinate
                start_y:        The top-left y-tile coordinate
                tile_width:     The number of tiles wide the image should be -
                                defaults to 5
                tile_height:    The number of tiles high the image should be -
                                defaults to 5
            Returns:
                A high-resolution Goole Map image.
        """

        start_x = kwargs.get('start_x', None)
        start_y = kwargs.get('start_y', None)
        tile_width = kwargs.get('tile_width', 5)
        tile_height = kwargs.get('tile_height', 5)

        # Check that we have x and y tile coordinates
        if start_x == None or start_y == None:
            start_x, start_y = self.getXY()

        # Determine the size of the image
        width, height = 256 * tile_width, 256 * tile_height

        # Create a new image of the size require
        map_img = Image.new('RGB', (width, height))

        for x in range(0, tile_width):
            for y in range(0, tile_height):
                url = 'http://mt0.google.com/vt/lyrs=s&hl=en&x=' + str(start_x + x) + '&y=' + str(
                    start_y + y) + '&z=' + str(
                    self._zoom)

                current_tile = str(x) + '-' + str(y)
                urllib.request.urlretrieve(url, current_tile)

                im = Image.open(current_tile)
                map_img.paste(im, (x * 256, y * 256))

                os.remove(current_tile)

        return map_img