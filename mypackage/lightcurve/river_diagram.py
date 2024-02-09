from mypackage.lightcurve.bin_lightcurve import bin_lightcurve
import numpy as np

def create_river_diagram(time:list, flux:list, period:float, cadence:float=None, fill_between:float=True):
    """Create a river diagram from a light curve and a given period

    Parameters
    ----------
    time : list
        _description_
    flux : list
        _description_
    period : float
        _description_

    Returns
    -------
    _type_
        return the river diagram, and a tuple of the new binned time, binned flux, and new cadence
    """
    new_time, new_flux, (std, mean_std, new_cadence, river_diagram_shape) = bin_lightcurve(time, flux, period=period, cadence=cadence, fill_between=fill_between)
    
    river_diagram = new_flux.reshape(river_diagram_shape)
    
    return river_diagram, (new_time, new_flux, new_cadence)


def create_river_diagram_with_qats_solution(qats_time, time, flux, period=False):    
    """Create a river diagram folded on the qats period, and add the qats solution on top of it
    Parameters
    ----------
    qats_time : _type_
        The time returned by qats
    time : _type_
        The time array of the light curve
    flux:
        The flux array of the light curve
    Returns
    -------
    _type_
        return the river diagram and the qats x and y scatter points solutions
    """
    if not period:
        period = np.mean(np.diff(qats_time))
    river_diagram, (folded_time, folded_flux, rd_cadence) = create_river_diagram(time, flux, period)
    
    transits_indices_binned = np.zeros(qats_time.shape[0], dtype=int)
    for i_t, t in enumerate(qats_time):
        transits_indices_binned[i_t] = np.abs(folded_time - t).argmin()
    
    transit_number = np.floor(transits_indices_binned / river_diagram.shape[1])
    transits_indices_binned = transits_indices_binned % river_diagram.shape[1]
    return river_diagram, transits_indices_binned, transit_number, (period, folded_time, rd_cadence, folded_flux)
    