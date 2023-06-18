import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
from rasterio.windows import Window
import rasterspace as rs
import numpy as np

tiles = np.load('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/tiles.npy')

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/distance.tif')

win = Window.from_slices((tiles[0,0,0], tiles[0,0,1]), (tiles[0,0,2], tiles[0,0,3]))

array = src.read(1, window=win).astype('float64')