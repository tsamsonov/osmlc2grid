# This function reads a subarray from TIFF file according to boundary coordinates

import numpy as np
from pyproj import Transformer

corners = np.array([
    [55.71671, 37.61184],
    [55.71344, 37.61982],
    [55.70895, 37.61182],
    [55.71346, 37.60385],
])

lon = [37.61184, 37.61982, 37.61182, 37.60385]
lat = [55.71671, 55.71344, 55.70895, 55.71346]

trs = Transformer.from_crs(4326, 32637)
x, y = trs.transform(lon, lat)

print((x, y))