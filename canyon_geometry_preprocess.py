import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
import rasterspace as rs
import numpy as np

wd = '/Volumes/Data/Spatial/OSM/CFO/2023-06-18'


euc = rs.euclidean_distance(array, 5.0)
with rasterio.open(f'{wd}/distance.tif', 'w', **profile) as dst:
    dst.write(euc, indexes = 1)

alloc = rs.euclidean_allocation(array, 5.0)
with rasterio.open(f'{wd}/allocation.tif', 'w', **profile) as dst:
    dst.write(alloc, indexes = 1)

tiles_width = rs.euclidean_width_tiles(euc, 5.0, 5, 5)
with open(f'{wd}/tiles_width.npy', 'wb') as dst:
    np.save(dst, tiles_width)

tiles_length = rs.raster_tiles(profile['height'], profile['width'], 5, 5, 2000)
with open(f'{wd}/tiles_length.npy', 'wb') as dst:
    np.save(dst, tiles_length)


band_names = [
    'Building height',
    'Predicted height',
    'Canyon width',
    'Canyon height',
    'Canyon direction',
    'Canyon length',
    'Sky view factor',
    'Worldcover',
    'OSM roads',
    'OSM industrial',
    'LCZ',
]

src = rasterio.open(f'{wd}/height_predicted.tif')
profile = src.profile

N = len(band_names)
profile.update(count=N, compress = 'lzw', dtype='int16')
dst = rasterio.open(f'{wd}/params.tif', 'w', **profile, BIGTIFF='YES')
# dst_meta = dst.meta
# dst_meta.update({"nodata": -32768})

idx = 1
array = src.read(1).astype('int16')
array[array == src.nodata] = -1
dst.write(array, indexes=idx)
dst.set_band_description(idx, band_names[idx-1])
src.close()

idx = 2
src = rasterio.open(f'{wd}/predicted.tif')
array = src.read(1).astype('int16')
array[array == src.nodata] = -1
# array -= 1
# array[array == -2] = -1

dst.write(array, indexes=idx)
dst.set_band_description(idx, band_names[idx-1])
src.close()

src = rasterio.open(f'{wd}/params_archive_2023-10-12.tif')
for idx in range(3, 9):
    array = src.read(idx-1).astype('int16')
    dst.write(array, indexes=idx)
    dst.set_band_description(idx, band_names[idx-1])
src.close()

idx = 9
src = rasterio.open(f'{wd}/roads_osm.tif')
array = src.read(1).astype('int16')
array[array == src.nodata] = -1
dst.write(array, indexes=idx)
dst.set_band_description(idx, band_names[idx-1])

idx = 10
src = rasterio.open(f'{wd}/industrial_osm.tif')
array = src.read(1).astype('int16')
array[array == src.nodata] = -1
dst.write(array, indexes=idx)
dst.set_band_description(idx, band_names[idx-1])

idx = 11
src = rasterio.open(f'{wd}/lcz.tif')
array = src.read(1).astype('int16')
array[array == src.nodata] = -1
dst.write(array, indexes=idx)
dst.set_band_description(idx, band_names[idx-1])
src.close()

# worldcover.close()

# NEW
dst.write(array, indexes=1)
for i in range(N):
    dst.set_band_description(i+1, band_names[i])
dst.close()

# SET MINUS

src = rasterio.open(f'{wd}/params_old.tif')
dst = rasterio.open(f'{wd}/params.tif', 'r+')

height = src.read(1).astype('int16')

for i in range(2, 4):

    array = src.read(i).astype('int16')
    array[array == 0] = -1
    array[height > 0] = -1

    dst.write(array, indexes=i)

dst.close()
