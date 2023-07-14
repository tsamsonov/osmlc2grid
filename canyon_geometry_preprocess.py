import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
import rasterspace as rs
import numpy as np

# src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/heights.tif')
# profile = src.profile
# array = src.read(1).astype('float64')
#
# euc = rs.euclidean_distance(array, 5.0)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)
#
# alloc = rs.euclidean_allocation(array, 5.0)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation.tif', 'w', **profile) as dst:
#     dst.write(alloc, indexes = 1)
#
# tiles_width = rs.euclidean_width_tiles(euc, 5.0, 5, 5)
# with open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles_width.npy', 'wb') as dst:
#     np.save(dst, tiles_width)

# tiles_length = rs.raster_tiles(profile['height'], profile['width'], 5, 5, 2000)
# with open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles_length.npy', 'wb') as dst:
#     np.save(dst, tiles_length)


# band_names = [
#     'Building height',
#     'Canyon width',
#     'Canyon height',
#     'Canyon direction',
#     'Canyon length',
#     'Sky view factor'
# ]

# N = len(band_names)
# profile.update(count=N, compress = 'lzw', dtype='int16')
# dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'w', **profile, BIGTIFF='YES')
# dst.write(array, indexes=1)
# for i in range(N):
#     dst.set_band_description(i+1, band_names[i])
#
# dst.close()

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params_old.tif')
dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'r+')

height = src.read(1).astype('int16')

for i in range(2, 4):

    array = src.read(i).astype('int16')
    array[array == 0] = -1
    array[height > 0] = -1

    dst.write(array, indexes=i)

src.close()
dst.close()
