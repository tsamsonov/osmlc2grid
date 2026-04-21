import math
import numpy as np
import geopandas as gpd
import rasterio

# pip install overturemaps
# overturemaps download --bbox=37.743104,55.647419,37.765763,55.659331 -f geojson --type=building -o maryino.geojson

# Параметр bbox легко получить на https://boundingbox.klokantech.com, установив выходной формат в CSV

FH  = 3.0   # высота этажа в метрах
RES = 2.0   # разрешение выходной сетки
PRJ = 32637 # выходная проекция (UTM 37N)

def floor_base(x, base=1):
    return base * math.floor(x / base)

def ceil_base(x, base=1):
    return base * math.ceil(x / base)

# читаем и проецируем данные
blds = gpd.read_file('maryino.geojson')
blds_utm = blds.to_crs(PRJ)

# для зданий с отсутствующей этажностью проставляем 2 этажа
blds_utm.fillna({'num_floors': 2}, inplace=True)

# вычисляем охват и размеры растра
box = blds_utm.geometry.total_bounds
west  = floor_base(box[0], RES) - RES
east  = ceil_base(box[2], RES)
south = floor_base(box[1], RES)
north = ceil_base(box[3], RES) + RES
nrow = int(1 + (north - south) / RES)
ncol = int(1 + (east - west) / RES)

# инициализируем пустой растр
transform = rasterio.transform.from_origin(west, north, RES, RES)
blds_raster = rasterio.open('maryino.tif', 'w', driver='GTiff',
                            height = nrow, width = ncol,
                            count=1, dtype=np.int16,
                            crs=PRJ, compress = 'lzw',
                            transform=transform)

# растрируем здания в numpy array
geom_value = ((geom, value) for geom, value in zip(blds_utm.geometry, FH * blds_utm['num_floors']))
arr = rasterio.features.rasterize(geom_value,
                         out_shape = (nrow, ncol),
                         transform = transform,
                         all_touched = True,
                         fill = 0,
                         dtype = np.int16)

# записываем результат
blds_raster.write(arr, 1)
blds_raster.close()