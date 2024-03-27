from mypackage.lightcurve.bin_lightcurve import bin_lightcurve, bin_lightcurve_faster
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



def create_river_diagram_faster(time:list, flux:list, period:float, cadence:float=None, fill_between:float=None):
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
    
    binned_time, binned_flux, (new_cadence, river_diagram_shape) = bin_lightcurve_faster(time, flux, period=period, cadence=cadence, fill_between=fill_between)
    
    river_diagram = binned_flux.reshape(river_diagram_shape)
    
    return river_diagram, (binned_time, binned_flux, new_cadence)


def transittime_to_riverdiagram_xy(transit_time, river_diagram, rd_time):    
    """Create a river diagram folded on the qats period, and add the qats solution on top of it
    Parameters
    ----------
    transit_time : _type_
        The time of transits
    river_diagram: _type_
        The river diagram matrix
    rd_time:
        The time returned by the creation of the river diagram
    Returns
    -------
    _type_
        return the x and y scatter points that corespond to the given river diagram
    """
    
    transits_indices_binned = np.zeros(transit_time.shape[0], dtype=int)
    for i_t, t in enumerate(transit_time):
        transits_indices_binned[i_t] = np.abs(rd_time - t).argmin()
    
    transit_number = np.floor(transits_indices_binned / river_diagram.shape[1])
    transits_indices_binned = transits_indices_binned % river_diagram.shape[1]
    return transits_indices_binned, transit_number