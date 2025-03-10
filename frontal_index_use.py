import os
import numpy as np
import rasterio
from matplotlib import pyplot
import frontal_index as fi


os.environ['USE_PYGEOS'] = '0'

src = rasterio.open('/Volumes/Work/__UCLIM/Kosheleva/mask_5m_small_blds.tif')
array = src.read(1).astype('float64')

flt = True

# subarray = array[485:555, 610:680] # XO SHI MIN
subarray = array[340:410, 475:545] # FIAN

if flt:
    mask = np.fromfunction(lambda i, j: abs(i-35) + abs(j-35) >= 35, subarray.shape, dtype=int)
    mask_idx = np.argwhere(mask)
    subarray[mask_idx[:, 0], mask_idx[:, 1]] = None

seq = np.array(list(range(0, 37))) * 5.0
FAI = []
FAIB = []
FAIS = []

for a in seq:
    FAI.append(fi.fai_buildings(subarray, a, cellsize=5.0))
    FAIB.append(fi.fai_blocking(subarray, a, cellsize=5.0))
    FAIS.append(fi.fai_walls(subarray, a, cellsize=5.0))

pyplot.imshow(subarray, cmap='bone')
pyplot.title('Высота здания, [м]')
pyplot.colorbar()
pyplot.show()

# pyplot.imshow(fi.mark_objects(subarray), cmap='bone')
# pyplot.show()

# exp = fi.exp(subarray, 90, cellsize=5.0)
# pyplot.imshow(exp, cmap='bone')
# pyplot.show()

fig = pyplot.figure()
ax = pyplot.axes()
ax.plot(seq, FAIS, label='FAI walls')
ax.plot(seq, FAI, label='FAI buildings')
ax.plot(seq, FAIB, label='FAI blocking')
ax.set_ylim(bottom=0)
pyplot.xlabel("Направление ветра, [°]")
ax.legend()
pyplot.title('Фронтальный индекс')
pyplot.show()
