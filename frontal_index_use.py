import os
import numpy as np
import rasterio
from matplotlib import pyplot
import frontal_index as fi


os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_small_blds.tif')
array = src.read(1).astype('float64')

subarray = array[485:555, 610:680]

mask = np.fromfunction(lambda i, j: abs(i-35) + abs(j-35) >= 35, subarray.shape, dtype=int)
mask_idx = np.argwhere(mask)
subarray[mask_idx[:, 0], mask_idx[:, 1]] = None

seq = np.array(list(range(0, 37))) * 5.0
FI = []

for a in seq:
    FI.append(fi.frontal_index(subarray, a, cellsize=5.0))

pyplot.imshow(subarray, cmap='bone')
pyplot.show()

fig = pyplot.figure()
ax = pyplot.axes()
ax.plot(seq, FI)
pyplot.show()