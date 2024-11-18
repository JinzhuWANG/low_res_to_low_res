import numpy as np
import rioxarray as rxr
import xarray as xr
from helper import bincount_avg


# Read id map
id_map = rxr.open_rasterio('data/NLUM_2010-11_mask_ID_map.tif', chunks='auto')
id_map = id_map.drop_vars('band').squeeze()       # remove the band dimension

# Read low resolution map
BIO_5km = rxr.open_rasterio('data/Arenophryne_xiphorhyncha_BCC-CSM2-MR_ssp370_2061-2080_AUS_5km_ClimSuit.tif', chunks='auto')

# Read high resolution map
lumap_30m = rxr.open_rasterio('data/lumap_2010.tiff', chunks='auto')  # 28 categories, -1 indicates no-agriculture, -9999 indicates no data
lumap_30m = lumap_30m.drop_vars('band').squeeze()       # remove the band dimension

# Convert lumap_30m to multiband map, each band represents a category
lumap_downsampled = []
for i in np.unique(lumap_30m.values):
    # skip no-agriculture and no data
    if i <0: 
        continue
    # create a mask for each category
    lumap_i = xr.DataArray(lumap_30m == i)
    # use bin_count to average the pixels in cell
    lumap_i = bincount_avg(lumap_i, id_map, BIO_5km.y.values, BIO_5km.x.values)
    # expand the dimension with lucode
    lumap_i = lumap_i.expand_dims({'lucode': [int(i)]})
    
    lumap_downsampled.append(lumap_i)

# Combine all bands
lumap_5km_multiband = xr.concat(lumap_downsampled, dim='lucode')


# Save the downsampled map
lumap_5km_multiband.rio.to_raster('data/lumap_5km_multiband.tiff', compress='LZW')