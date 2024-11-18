import numpy as np
import xarray as xr
from rasterio.enums import Resampling


def get_id_map_by_upsample_reproject(low_res_map, high_res_map):
    """
    Upsamples and reprojects a low-resolution map to match the resolution and 
    coordinate reference system (CRS) of a high-resolution map.
    
    Parameters:
        low_res_map (2D xarray.DataArray): The low-resolution map to upsample and reproject. Should have CRS and affine transformation.
        high_res_map (2D xarray.DataArray): The high-resolution map to match the resolution and CRS to. Should have CRS and affine transformation.
    
    Returns:
        xarray.DataArray: The upsampled and reprojected map with the same CRS and resolution as the high-resolution map.
    """
    low_res_id_map = np.arange(low_res_map.size).reshape(low_res_map.shape)
    low_res_id_map = xr.DataArray(
        low_res_id_map, 
        dims=['y', 'x'], 
        coords={'y': low_res_map.coords['y'], 'x': low_res_map.coords['x']})
    
    low_res_id_map = low_res_id_map.rio.write_crs(low_res_map.rio.crs)
    low_res_id_map = low_res_id_map.rio.write_transform(low_res_map.rio.transform())
    low_res_id_map = low_res_id_map.rio.reproject_match(
        high_res_map, 
        Resampling = Resampling.nearest, 
        nodata=low_res_map.size + 1).chunk('auto')
    
    return low_res_id_map



def bincount_avg(weight_arr, bin_arr, y, x):
    """
    Calculate the average value of each bin based on the weighted values.

    Parameters:
    - weight_arr (xarray.DataArray): Array of weighted values.
    - bin_arr (xarray.DataArray): Array of bin values.
    - y (numpy.ndarray): Array of y coordinates.
    - x (numpy.ndarray): Array of x coordinates.

    Returns:
    - xarray.DataArray: Array of average values for each bin.

    """
    
    # Get the size and shape of the output array
    out_shape = (y.size, x.size)
    bin_size = int(y.size * x.size)
    
    # Flatten arries
    bin_flatten = bin_arr.values.flatten()
    weights_flatten = weight_arr.values.flatten()
    
    bin_occ = np.bincount(bin_flatten, minlength=bin_size)
    bin_sum = np.bincount(bin_flatten, weights=weights_flatten, minlength=bin_size)
  

    # Calculate the average value of each bin, ignoring division by zero (which will be nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        bin_avg = (bin_sum / bin_occ).reshape(out_shape).astype(np.float32)
        bin_avg = np.nan_to_num(bin_avg)
        

    # Expand the dimensions of the bin_avg array to match the original weight_arr
    bin_avg = xr.DataArray(bin_avg, dims=('y', 'x'), coords={'y': y, 'x': x})
     
    return bin_avg
