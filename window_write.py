import os
os.environ['USE_PYGEOS'] = '0'

import rasterio
from rasterio.windows import Window
import numpy as np

win = Window.from_slices((100, 200), (100, 200))

src = rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/euc_small.tif', 'r+')
distance = src.read(1, window=win).astype('float64')
scaled = distance * 10
src.write(scaled, indexes = 1, window=win)