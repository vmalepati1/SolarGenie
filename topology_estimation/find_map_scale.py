import math

def find_map_scale_from_area(latitude, desired_area):
    ground_scale2 = math.pow((math.cos(latitude * math.pi / 180) * 2 * math.pi * 6378137), 2)
    return round(0.5 * math.log2(ground_scale2 / desired_area))