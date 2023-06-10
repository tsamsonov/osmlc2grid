import os
import rasterio
import rasterspace as rs
import numpy as np
from matplotlib import pyplot


os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_small_blds.tif')
array = src.read(1).astype('float64')

rescaled = rs.rescale(array)
euc = rs.euclidean_transform(array)

pyplot.imshow(euc[0, :, :], cmap='bone')
pyplot.show()

pyplot.imshow(euc[1, :, :], cmap='bone')
pyplot.show()
