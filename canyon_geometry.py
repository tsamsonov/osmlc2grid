import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
from rasterio.windows import Window
import rasterspace as rs
import numpy as np
import time

tiles = np.load('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles.npy')
win = Window.from_slices((tiles[2,3,0], tiles[2,3,1]), (tiles[2,3,2], tiles[2,3,3]))

dist_src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif')
distance = dist_src.read(1, window=win).astype('float64')

alloc_src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation.tif')
allocation = alloc_src.read(1, window=win).astype('float64')

start = time.time()
params = rs.euclidean_width_params_split2(distance, allocation, 5.0, 5, 5)
end = time.time()
print(end - start)

band_names = [
    'Dominant pixel',
    'Dominant distance',
    'Canyon width',
    'Canyon height',
    'Canyon H/W ratio',
    'Building distance',
    'Building height',
]

profile = dist_src.profile
profile.update(count=7, compress='lzw', dtype='float64')
with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'w+', **profile) as dst:
    for i in range(7):
        dst.write(params[i,:,:], indexes=i+1, window=win)
        dst.set_band_description(i+1, band_names[i])