from geopandas.tools import geocode

g = geocode(["901 Highbury Ln, Marietta, GA"], timeout=5.0)
lat = g.geometry[0].y
long = g.geometry[0].x
print(lat)
print(long)