import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
import rasterspace as rs
import numpy as np

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/heights.tif')
profile = src.profile
array = src.read(1).astype('float64')

euc = rs.euclidean_distance(array, 5.0)
with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif', 'w', **profile) as dst:
    dst.write(euc, indexes = 1)

alloc = rs.euclidean_allocation(array, 5.0)
with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation.tif', 'w', **profile) as dst:
    dst.write(alloc, indexes = 1)

tiles = rs.euclidean_width_tiles(euc, 5.0, 5, 5)
with open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles.npy', 'wb') as dst:
    np.save(dst, tiles)

band_names = [
    'Building height',
    'Canyon width',
    'Canyon height',
    # 'Dominant pixel',
    # 'Dominant distance',
    # 'Canyon H/W ratio',
    # 'Building distance',
    # 'Building height',
]

N = len(band_names)
profile.update(count=N, compress = 'lzw', dtype='int16')
dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'w', **profile, BIGTIFF='YES')
dst.write(array, indexes=1)
for i in range(N):
    dst.set_band_description(i+1, band_names[i])

dst.close()

# dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'r+')
# dst.write(array, indexes=1)
# dst.set_band_description(1, 'Building height')
# dst.close()
