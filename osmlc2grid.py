import os
import rasterio
import rasterspace as rs
import time
import numpy as np
from matplotlib import pyplot

os.environ['USE_PYGEOS'] = '0'

# src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/blds_5m.tif')
src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_ttk.tif')
profile = src.profile
# profile.update(count = 1, compress = 'lzw')
#
array = src.read(1).astype('float64')
#
# start = time.time()
# euc = rs.euclidean_distance(array, 5.0)
# end = time.time()
# print(end - start)
#
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/euc_small.tif', 'w', **profile) as dst:
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
# euc = rs.euclidean_width_split(array, 5.0, 4, 3)
# end = time.time()
# print(end - start)
#
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_split.tif', 'w', **profile) as dst:
#     dst.write(euc, indexes = 1)

# pyplot.imshow(euc[0, :, :], cmap='bone')
# pyplot.show()

# start = time.time()
# width = rs.euclidean_width_params(array, 5.0)
# end = time.time()
# print(end - start)
#
# profile.update(count = 6, compress = 'lzw')
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_params_ttk.tif', 'w', **profile) as dst:
#     dst.write(width)

start = time.time()
width = rs.euclidean_width_params_split(array, 5.0, 3, 4)
end = time.time()
print(end - start)

profile.update(count = 6, compress = 'lzw')
with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/width_params_ttk_split.tif', 'w', **profile) as dst:
    dst.write(width)

# start = time.time()
# center = rs.euclidean_centrality(array, 5.0)
# end = time.time()
# print(end - start)

# pyplot.imshow(width[1, :, :], cmap='bone')
# pyplot.show()

# profile.update(count = 1, compress = 'lzw')
# with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/centr_small.tif', 'w', **profile) as dst:
#     dst.write(center, indexes = 1)
