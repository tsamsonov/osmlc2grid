import rasterio
from rasterio.windows import Window
import rasterspace as rs
import numpy as np

wd = '/Volumes/Work/UCLIM/Vlada'

dist_src = rasterio.open(f'{wd}/distance.tif')
dist = dist_src.read(1).astype('float64')

alloc_src = rasterio.open(f'{wd}/allocation_correct.tif')
alloc = alloc_src.read(1).astype('float64')

width_src = rasterio.open(f'{wd}/canyon_params_raw.tif')
width = width_src.read(2).astype('float64')

params = rs.euclidean_length_params(dist, alloc, width, 6.0, 5000.0, 5.0)