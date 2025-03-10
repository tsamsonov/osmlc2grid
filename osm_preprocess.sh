osmium extract -p MSK_0045_bbox_UTM_bbox_WGS_bbox.geojson central-fed-district-latest.osm.pbf -o moscow_0045.osm.pbf

osmium tags-filter -o buildings.osm.pbf moscow_0045.osm.pbf awr/building awr/building:part  --overwrite

ogr2ogr buildings.gpkg buildings.osm.pbf -oo CONFIG_FILE=buildings.ini -t_srs EPSG:32637 -clipdst MSK_0045_bbox_UTM_bbox.geojson

gdal_rasterize -a height -ot Int16 -tr 5 5 -l multipolygons buildings.gpkg heights.tif -sql "SELECT * FROM multipolygons order by height"

gdal_rasterize -a id -ot Int32 -tr 5 5 -l multipolygons buildings.gpkg fids.tif -sql "SELECT *, ROW_NUMBER() OVER(ORDER BY height) AS id FROM multipolygons"

gdalbuildvrt worldcover.vrt ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000/*.tif

gdalwarp -cutline MSK_0045_bbox_UTM_bbox.geojson -crop_to_cutline -t_srs EPSG:32637 -tr 5 5 -r near worldcover.vrt worldcover.tif

gdalwarp -te 296823.0009229568531737 6069879.4020483121275902 509188.0009229568531737 6277829.4020483121275902 -t_srs EPSG:32637 -tr 5 5 -r near worldcover.vrt worldcover.tif