import os
import rasterio
import rasterspace as rs
import numpy as np
from matplotlib import pyplot


os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_small_blds.tif')
array = src.read(1).astype('float64')

rescaled = rs.rescale(array)
euc = rs.euclidean_transform(array, 5.0)

pyplot.imshow(euc[0, :, :], cmap='bone')
pyplot.show()

width = rs.euclidean_width(euc, 5.0)

pyplot.imshow(width[1, :, :], cmap='bone')
pyplot.show()

profile = src.profile
profile.update(count = 4, compress = 'lzw')

with rasterio.open('/Users/tsamsonov/GitHub/osmlc2grid/data/output.tif', 'w', **profile) as dst:
    dst.write(width)