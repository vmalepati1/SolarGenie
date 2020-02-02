from geopandas.tools import geocode

g = geocode(["3280 Sewell Mill Rd, Marietta, GA 30062"], timeout=5.0)
lat = g.geometry[0].y
long = g.geometry[0].x
print(lat)
