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



def transittime_to_riverdiagram_xy(transit_time: np.ndarray, t0: float, river_diagram: np.ndarray, river_diagram_folding_period: float):    
    """Convert the transit time into x and y river diagram points
    ----------
    transit_time : np.ndarray
        The time of transits.
    t0 : float
        The first time of the light curve.
    river_diagram: np.ndarray
        The river diagram matrix.
    river_diagram_folding_period:
        The time returned by the creation of the river diagram.
    Returns
    -------
    transit_time_rd : np.ndarray
        return the river diagram x points.
    transit_number_rd : np.ndarray
        return the river diagram y points.
    """
    
    transit_time_rd = np.floor((transit_time - t0) % river_diagram_folding_period / river_diagram_folding_period * river_diagram.shape[1]).astype(int)
    transit_number_rd = np.floor((transit_time - t0) / river_diagram_folding_period).astype(int)
    
    return transit_time_rd, transit_number_rd