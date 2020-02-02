import osmnx as ox
import math

def get_building_footprint(address, sq_meters):
    search_radius = math.sqrt(sq_meters)
    footprint = ox.footprints_from_address(address, search_radius)
    return footprint.geometry.iloc[0]