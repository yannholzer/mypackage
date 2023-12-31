from mypackage.lightcurve.bin_lightcurve import bin_lightcurve

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
        return the river diagram, and a tuple of the new binned light curve, cadence
    """
    new_time, new_flux, (std, mean_std, new_cadence, river_diagram_shape) = bin_lightcurve(time, flux, period=period, cadence=cadence, fill_between=fill_between)
    
    river_diagram = new_flux.reshape(river_diagram_shape)
    
    return river_diagram, (new_time, new_flux, new_cadence)
    