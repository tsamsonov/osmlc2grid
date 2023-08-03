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
FAI = []
FAIB = []
FAIS = []

# exp = fi.frontal_index_surface(subarray, 90, cellsize=5.0)
# pyplot.imshow(exp, cmap='bone')
# pyplot.show()

for a in seq:
    FAI.append(fi.frontal_index(subarray, a, cellsize=5.0))
    FAIB.append(fi.frontal_index_blocking(subarray, a, cellsize=5.0))
    FAIS.append(fi.frontal_index_surface(subarray, a, cellsize=5.0))

# pyplot.imshow(subarray, cmap='bone')
# pyplot.show()
#
# pyplot.imshow(fi.mark_objects(subarray), cmap='bone')
# pyplot.show()

fig = pyplot.figure()
ax = pyplot.axes()
ax.plot(seq, FAI, label='FAI non-blocking')
ax.plot(seq, FAIS, label='FAI surface')
ax.plot(seq, FAIB, label='FAI blocking')
ax.legend()
pyplot.show()
