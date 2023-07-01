import os
import rasterio
import rasterspace as rs
import time
import numpy as np
from matplotlib import pyplot

os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance_ttk.tif')
distance = src.read(1).astype('float64')

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation_ttk.tif')
height = src.read(1).astype('float64')

profile = src.profile
profile.update(count = 3, compress = 'lzw', dtype='int16')

length = rs.euclidean_length_params(distance, height, 0.25, 5000, 5.0)

dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/length_fast.tif', 'w', **profile, BIGTIFF='YES')
dst.write(length)



# start = time.time()
# src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/heights.tif')
# src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/blds_5m.tif')
# src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/fids_ttk.tif')
# profile = src.profile
# profile.update(count = 1, compress = 'lzw')
#
# array = src.read(1).astype('float64')
# end = time.time()
# print(end - start)


# start = time.time()
# euc = rs.euclidean_distance(array, 5.0)
# end = time.time()
# print(end - start)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance_ttk.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)


# start = time.time()
# euc = rs.euclidean_antidistance(array, 5.0)
# end = time.time()
# print(end - start)
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/antidistance_ttk.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)
#
#
# start = time.time()
# euc = rs.euclidean_width(array, 5.0)
# end = time.time()
# print(end - start)
#
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)


# start = time.time()
# euc = rs.euclidean_width_split(array, 5.0, 3, 4, True)
# end = time.time()
# print(end - start)
#
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_split_ttk.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)

# pyplot.imshow(euc[0, :, :], cmap='bone')
# pyplot.show()

# band_names = [
#     'Dominant pixel',
#     'Canyon width',
#     'Canyon height',
#     'Canyon H/W ratio',
#     'Building distance',
#     'Building height',
# ]

# start = time.time()
# width = rs.euclidean_width_params(array, 5.0)
# end = time.time()
# print(end - start)

# profile.update(count = 6, compress = 'lzw')
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/width.tif', 'w', **profile) as dst:
# # with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_params.tif', 'w', **profile) as dst:
#     dst.write(width)
#     for i in range(6):
#         dst.set_band_description(i+1, band_names[i])

# start = time.time()
# width = rs.euclidean_width_params_split(array, 5.0, 20, 20)
# end = time.time()
# print(end - start)
#
# profile.update(count=6, compress='lzw')
# with rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/width.tif', 'w', **profile) as dst:
# # with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_params_split.tif', 'w', **profile) as dst:
#     dst.write(width)
#     for i in range(6):
#         dst.set_band_description(i+1, band_names[i])

# start = time.time()
# center = rs.euclidean_centrality(array, 5.0)
# end = time.time()
# print(end - start)

# pyplot.imshow(width[1, :, :], cmap='bone')
# pyplot.show()

# profile.update(count = 1, compress = 'lzw')
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/centr_small.tif', 'w', **profile) as dst:
#     dst.write(center, indexes = 1)
