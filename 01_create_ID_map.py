import numpy as np
import rioxarray as rxr
from helper import get_id_map_by_upsample_reproject




# Get maps with different resolutions
NLUM_30m = rxr.open_rasterio('data/NLUM_2010-11_mask.tif', chunks='auto').astype(np.bool_)
BIO_5km = rxr.open_rasterio('data/Arenophryne_xiphorhyncha_BCC-CSM2-MR_ssp370_2061-2080_AUS_5km_ClimSuit.tif', chunks='auto')

# The dimension of 'band' is redundant, so we remove it
NLUM_30m = NLUM_30m.drop_vars('band').squeeze()
BIO_5km = BIO_5km.drop_vars('band').squeeze()

# Make sure both maps have a valid CRS
'''The BIO_5km raster does not have a CRS. But we know that it is in the same CRS as the NLUM_30m raster.'''
BIO_5km = BIO_5km.rio.write_crs(NLUM_30m.rio.crs)


# Get the ID map
id_map = get_id_map_by_upsample_reproject(BIO_5km, NLUM_30m)


# Save the ID map, open it in ArcGIS/QGIS, and check it with the original NLUM map
id_map.rio.to_raster('data/NLUM_2010-11_mask_ID_map.tif', compress='LZW')
