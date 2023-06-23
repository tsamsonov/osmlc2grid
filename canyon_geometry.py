import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
from rasterio.windows import Window
import rasterspace as rs
import numpy as np
import time

tiles = np.load('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles.npy')

dim = np.shape(tiles)

print(tiles)

start = time.time()

for i in range(dim[0]):
    for j in range(dim[1]):

        print(f'\nPROCESSING TILE {i}, {j}\n')

        win_write = Window.from_slices((tiles[i, j, 0], tiles[i, j, 1]), (tiles[i, j, 2], tiles[i, j, 3]))
        win_read  = Window.from_slices((tiles[i, j, 4], tiles[i, j, 5]), (tiles[i, j, 6], tiles[i, j, 7]))

        row1 = tiles[i, j, 0] - tiles[i, j, 4]
        row2 = tiles[i, j, 1] - tiles[i, j, 4]
        col1 = tiles[i, j, 2] - tiles[i, j, 6]
        col2 = tiles[i, j, 3] - tiles[i, j, 6]

        dist_src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif')
        distance = dist_src.read(1, window=win_read).astype('float64')
        dist_src.close()

        alloc_src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/allocation.tif')
        allocation = alloc_src.read(1, window=win_read).astype('float64')
        alloc_src.close()

        params = rs.euclidean_width_params_split(distance, allocation, 5.0, 3, 3)

        dst = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'r+')
        for k in range(2):
            dst.write(params[k, row1:row2, col1:col2], indexes=k+2, window=win_write)
        dst.close()

end = time.time()

print(end - start)