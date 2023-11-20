import numpy as np
from typing import Tuple

def bin_lightcurve(time:list, flux:list, cadence:float=None, period:float=None, n_bins=None, fill_between=True) -> Tuple[list, list, Tuple[list, float, Tuple[float, float]]]:
    """Bin a light curve consisting in a time and a flux array with a chosen cadence or period. The chosen period make sure the light curve can be folded exactly on this periodicity.

    Parameters
    ----------
    time : list
        A list containing the time serie
    flux : list
        A list containing the flux coresponding to each time
    cadence : float, optional
        The chosen cadence to bin the new light curve
    period : float, optional
        The chosen period that will define a new binning of the light curve

    Returns
    -------
    Tuple[list, list, Tuple[float, Tuple[int, int]]]
        Return: 
        - new binned time, 
        - new binned flux, 
        - std of binned flux,
        - new cadence,
        - the shape of a river diagram folded on the given period if given, or empty shape otherwise.
    """
    
    n_rows = None
    n_columns = None
    if n_bins is not None:
        new_binned_time = np.linspace(time[0], time[-1], n_bins, endpoint=False)
        cadence = np.mean(np.diff(new_binned_time))
        new_binned_time += cadence/2
   
    
    else:
        if cadence is None:
            cadence = np.mean(np.diff(time))
            
        n_bins = round((time[-1]-time[0])/cadence)
        
        if period is None:
            
            new_binned_time = np.arange(time[0] + cadence/2, time[0] + n_bins*cadence+cadence/2, cadence)
         
        else:
            n_bin_in_period = np.floor(period / cadence).astype(int)
            cadence = period / n_bin_in_period
            n_transit = (time[-1] - time[0]) / period   
            n_rows = np.ceil(n_transit).astype(int)
            n_columns = n_bin_in_period
            n_bins = n_rows*n_columns
            new_binned_time = np.arange(time[0] + cadence/2, time[0] + n_bins*cadence + cadence/2, cadence)[:int(n_rows*n_columns)] # hard fix for dimension bug, TODO
        
    mean_flux = np.mean(flux)    
    sigma_flux = np.std(flux)

    new_binned_flux = np.ones_like(new_binned_time)*mean_flux
    std_binned_flux = np.zeros_like(new_binned_time)
    for i in range(n_bins):
        bin_start_time = time[0] + i * cadence
        bin_end_time = time[0] +(i+1) * cadence
        indices_in_bin = np.where((time >= bin_start_time) & (time < bin_end_time))[0]    
        if indices_in_bin.size > 0:
            new_binned_flux[i] = np.mean(flux[indices_in_bin])
            std_binned_flux[i] = np.std(flux[indices_in_bin])
        else:
            if fill_between:
                new_binned_flux[i] = np.random.normal(mean_flux, sigma_flux)
                std_binned_flux[i] = sigma_flux
    
    
    river_diagram_shape = (n_rows, n_columns)

    return new_binned_time, new_binned_flux, (std_binned_flux, cadence, river_diagram_shape)






