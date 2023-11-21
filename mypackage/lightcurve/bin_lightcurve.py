import numpy as np
from typing import Tuple

def bin_lightcurve(time:list, flux:list, cadence:float=None, period:float=None, n_bins=None, fill_between=np.nan) -> Tuple[list, list, Tuple[list, float, Tuple[float, float]]]:
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
        bins = np.linspace(time[0], time[-1], n_bins, endpoint=False)
        cadence = np.mean(np.diff(bins))
        bins += cadence
        binned_time = bins - cadence/2
        
        
   
    
    else:
        if cadence is None:
            cadence = np.median(np.diff(time)) # median if gap
            
        n_bins = round((time[-1] - time[0])/cadence)
        if period is None:
            bins = np.linspace(time[0], time[-1], n_bins, endpoint=False)
            cadence = np.mean(np.diff(bins))
            bins += cadence
            binned_time = bins - cadence/2
        else:
            n_bin_in_period = np.floor(period / cadence).astype(int)
            cadence = period / n_bin_in_period
            n_transit = (time[-1] - time[0]) / period   
            n_rows = np.ceil(n_transit).astype(int)
            n_columns = n_bin_in_period
            n_bins = n_rows*n_columns
            max_time = cadence * n_bins + time[0]
            bins = np.linspace(time[0], max_time, n_bins)
            cadence = np.mean(np.diff(bins))
            bins += cadence
            binned_time = (bins - cadence/2)[:int(n_rows*n_columns)]
            # hard fix for dimension bug, TODO
        
       
        
    mean_flux = np.mean(flux)    
    sigma_flux = np.std(flux)
    
    digitized = np.searchsorted(bins, time, side='left') 
    binned_flux = np.zeros_like(bins)
    std_binned_flux = np.zeros_like(bins)
    mean_std_binned_flux = np.zeros_like(bins)



    
    for i in range(0, len(bins)):
        window = np.where(digitized == i)[0]
        if window.size > 0:
            binned_flux[i] = np.mean(flux[window])
            std_binned_flux[i] = np.std(flux[window])
            mean_std_binned_flux[i] = np.std(flux[window])/np.sqrt(window.size)
        else:
            if fill_between is True:
                binned_flux[i] = np.random.normal(mean_flux, sigma_flux)
                std_binned_flux[i] = sigma_flux
            else:
                binned_flux[i] = fill_between
                std_binned_flux[i] = fill_between
                
    
    
    river_diagram_shape = (n_rows, n_columns)

    return binned_time, binned_flux, (std_binned_flux, mean_std_binned_flux, cadence, river_diagram_shape)




