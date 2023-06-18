# Import modules

import os
os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg
from rasterio.plot import show
from numpy import int16

# Extract buildings

# osmium extract -p MSK_0045_bbox.geojson central-fed-district-latest.osm.pbf -o moscow_0045.osm.pbf
# osmium tags-filter -o buildings.osm.pbf moscow_0045.osm.pbf awr/building awr/building:part  --overwrite
# ogr2ogr buildings.gpkg buildings.osm.pbf -oo CONFIG_FILE=buildings.ini -t_srs EPSG:32637
# gdal_rasterize -a height -ot Int16 -tr 5 5 -l multipolygons buildings.gpkg heights.tif -sql "SELECT * FROM multipolygons order by height"