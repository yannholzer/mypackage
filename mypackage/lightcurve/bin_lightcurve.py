import numpy as np
from typing import Tuple

def bin_lightcurve(time:list, flux:list, cadence:float=None, period:float=None) -> Tuple[list, list, float, float]:
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
    Tuple[list, list, float, Tuple[int, int]]
        Return: 
        - new binned time, 
        - new binned flux, 
        - new cadence
        - the shape of a river diagram folded on the given period if given, or empty shape otherwise.
    """
    if period is None and cadence is None:
        raise ValueError(
            "Please give at least a period or a cadence"
        )
    if period is None:
        n_bin = round((time[-1]-time[0])/cadence)
        n_rows = 0
        n_columns = 0
        new_binned_time = np.arange(time[0] + cadence/2, time[-1]+cadence/2, cadence)
    
    else:
        if cadence is None:
            cadence = np.abs(time - np.roll(time, 1)).min()
        n_bin_in_period = period / cadence
        n_transit = (time[-1] - time[0]) / period   
        n_rows = np.ceil(n_transit).astype(int)
        n_columns = np.floor(n_bin_in_period).astype(int)
        cadence = period / n_columns
        n_bin = n_rows*n_columns
        new_binned_time = np.arange(time[0] + cadence/2, time[0]+ n_bin*cadence + cadence, cadence)[:int(n_rows*n_columns)] # hard fix for dimension bug, TODO
        
    
    print(n_bin, new_binned_time.shape[0])

    new_binned_flux = np.ones_like(new_binned_time)*np.mean(flux)
    std_binned_flux = np.zeros_like(new_binned_time)
    
    
    for i in range(n_bin):
        bin_start_time = i * cadence
        bin_end_time = (i+1) * cadence
        indices_in_bin = np.where((time >= bin_start_time) & (time < bin_end_time))[0]    
        if indices_in_bin.size > 0:
            new_binned_flux[i] = np.mean(flux[indices_in_bin])
            std_binned_flux[i] = np.std(flux[indices_in_bin])
        else:
            new_binned_flux[i] = np.nan
            std_binned_flux[i] = np.nan
    
    
    river_diagram_shape = (n_rows, n_columns)

    return new_binned_time, new_binned_flux, std_binned_flux, cadence, river_diagram_shape