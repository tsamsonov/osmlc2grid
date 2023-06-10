import os
import rasterio
import rasterspace as rs
import numpy as np
from matplotlib import pyplot


os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_small_blds.tif')
array = src.read(1).astype('float64')

rescaled = rs.rescale(array)
euc = rs.euclidean_distance(array)

pyplot.imshow(euc, cmap='bone')
pyplot.show()
