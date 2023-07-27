# This function reads a subarray from TIFF file according to boundary coordinates
import shapely
from pyproj import Transformer
import rasterio.mask
import rasterio
from matplotlib import pyplot

def read_masked(src, lon, lat, crs=4326):
    trs = Transformer.from_crs(crs, src.crs)
    x, y = trs.transform(lat, lon)
    x.append(x[0])
    y.append(y[0])
    coords = list(zip(x, y))
    mask = [shapely.Polygon(coords)]
    subarray, transform = rasterio.mask.mask(src, mask, crop=True, filled=False)
    return subarray

lon = [37.61184, 37.61982, 37.61182, 37.60385]
lat = [55.71671, 55.71344, 55.70895, 55.71346]

src = rasterio.open('/Volumes/Data/Spatial/OSM/CFO/2023-06-18/params.tif', 'r')
subarray = read_masked(src, lon, lat)

pyplot.imshow(subarray[0, :, :], cmap='bone')
pyplot.show()


