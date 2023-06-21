import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
import rasterspace as rs
import numpy as np

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/heights.tif')
profile = src.profile
array = src.read(1).astype('float64')

# euc = rs.euclidean_distance(array, 5.0)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)
#
# tiles = rs.euclidean_width_tiles(euc, 5.0, 5, 5)
# with open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles.npy', 'wb') as dst:
#     np.save(dst, tiles)

# alloc = rs.euclidean_allocation(array, 5.0)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation.tif', 'w', **profile) as dst:
#     dst.write(alloc, indexes = 1)

profile.update(count = 7, compress = 'lzw', dtype='float64')
band_names = [
    'Dominant pixel',
    'Dominant distance',
    'Canyon width',
    'Canyon height',
    'Canyon H/W ratio',
    'Building distance',
    'Building height',
]
with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'w', **profile, BIGTIFF='YES') as dst:
    for i in range(7):
        dst.set_band_description(i+1, band_names[i])