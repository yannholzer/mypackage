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
        new_binned_time = np.arange(time[0], time[-1] + cadence/2, cadence)
        new_binned_flux = np.ones_like(new_binned_time)*np.mean(flux)
        Nrows = 0
        new_bin = 0
        new_cadence = cadence
    
    else:
        Ntransit = (time[-1] - time[0]) / period
        Nrows = np.ceil(Ntransit).astype(int)
        Nbin_in_period = period / cadence
        new_bin = np.ceil(Nbin_in_period).astype(int)
        new_cadence = period / new_bin
        new_binned_time = np.arange(time[0], Nrows*period+time[0], new_cadence)[:int(new_bin*Nrows)]
        new_binned_flux = np.ones_like(new_binned_time)*np.mean(flux)

    right = np.searchsorted(time, new_binned_time, "right")
    
    prev = 0
    for i in range(new_binned_time.shape[0]):
        if prev != right[i]:    
            new_binned_flux[i] = np.mean(flux[prev:right[i]])
            prev = right[i]

    river_diagram_shape = (Nrows, new_bin)

    return new_binned_time, new_binned_flux, river_diagram_shape, new_cadence