import numpy as np
from typing import Tuple
import os

def generate_light_curve(period:float=None, duration:float=None, epoch:float=None, observation_time:float = None, cadence:float=None, transit_depth_fraction:float=None, sigma:float=None, variation:Tuple[float, float, float]=None) -> Tuple[list, list]:
    """Synthetic light curve generator. The time arguments are in unit of Days.
    Parameters
    ----------
    period : float, optional
        The periodicity in days of the transit within the light curve, if None a random value is selected between 5 and 10 days
    duration : float, optional
        The duration in days of the transit whin the light curve, if None a random value is selected between 0.05 and 0.1 days
    epoch : float, optional
        The epoch in days of the transit, if None a random value is selected between 0 and the period.
    observation_time : float, optional
        The total duration of observation of the light curve, if None a duration of 4 years is selected.
    cadence : float, optional
        the cadence in days of the observation, if None a cadence of 29.424 minutes is selected.
    transit_depth_fraction : float, optional
        The depth of the transit as a fraction of one, if None a depth of 0.995 is selected.
    sigma : float, optional
        The normal standard deviation of the noise injected in the transit , if None the deviation is equal to the depth of the transit.
    variation : tuple, optional
        A tuple containing the amplitude, frequency and phase of a timing variation of the transits time, if default None, random parameters are selected, if False, no variation in the transits time.
    
    Returns
    -------
    Tuple[list, list]
        Return the time and the flux of the generated light curve.
    """
    if period is None:
        period = np.random.uniform(5, 10)
    if observation_time is None:
        observation_time = 4*365
    if cadence is None:
        cadence = 29.4/(60*24)
    if duration is None:
        duration = np.random.uniform(0.05, 0.1)
    if epoch is None:
        epoch = np.random.uniform(0, period)
    if transit_depth_fraction is None:
        transit_depth_fraction = 0.995

    time = np.arange(0, observation_time, cadence)
    flux = np.ones_like(time)
    
    transits_time = np.arange(epoch, observation_time, period)

    if variation is False:
        variation = (0, 1, 0)
        left_points = np.searchsorted(time, transits_time, "left")
        right_points = np.searchsorted(time, transits_time+duration, "right")
        
    elif variation is None:
        variation_amplitude = np.random.uniform(0, duration)
        variation_period = np.random.uniform(5, 10)*observation_time
        variation_phase = np.random.uniform(0, 2*np.pi)
        variation = (variation_amplitude, variation_period, variation_phase)
    
    try:
        variation_amplitude, variation_period, variation_phase = variation
        time_variation = variation_amplitude*np.sin(2*np.pi*transits_time/variation_period + variation_phase)
        left_points = np.searchsorted(time, transits_time + time_variation, "left")
        right_points = np.searchsorted(time, transits_time + time_variation + duration, "right")
    
    except Exception as e:
        raise(e,"error unpacking variation parameters")
        
    
    for i in range(transits_time.shape[0]):
        flux[left_points[i]:right_points[i]] *= transit_depth_fraction

    if sigma is None:
        sigma = (1- transit_depth_fraction)
        flux += np.random.normal(0, sigma, time.shape)
    elif sigma is False:
        pass
    else:
        flux += np.random.normal(0, sigma, time.shape)
    
    return time, flux


def create_npy_lightcurve_dataset(
    n_data:int, 
    path_to_export:str,
    data_name:str,
    compress:bool=False,
    subfolder:bool=False,
    data_format:str="lightcurve",
    name_period_range_format:str=True,
    period_range:Tuple[float, float]=(5, 10), 
    duration_range:Tuple[float, float]=(0.05, 0.1), 
    observation_time:float=4*365, 
    transit_depth_fraction:float=0.995, 
    cadence:float=29.4/(60*24), 
    epoch:float=None, 
    sigma:float=0.005, 
    variation_parameters:float=None
    ):
    data_formats = ["lightcurve", "river_diagram"]
    
    if data_format not in data_formats:
        raise(f"data format not valid. please select a format from {data_formats}")
    
    period = np.random.uniform(*period_range, n_data)
    duration = np.random.uniform(*duration_range, n_data)
    
    
    path_root = os.path.join(path_to_export, data_format, "raw")
    os.makedirs(path_root, exist_ok=True)
    path_data = os.path.join(path_root, data_name)
    os.mkdir(path_data)
    for n in range(n_data):
          
        time, flux = generate_light_curve(period[n], duration[n], epoch, observation_time, cadence, transit_depth_fraction, sigma ,variation_parameters)
        
        if name_period_range_format:
            name = f"lc{str(n).zfill(len(str(n_data-1)))}_{str(period_range[0]).replace('.', 'p')}_{str(period_range[1]).replace('.', 'p')}"
        else:
            name = f"lc{str(n).zfill(len(str(n_data-1)))}_{str(period_range[0]).replace('.', 'p')}-{str(period_range[1]).replace('.', 'p')}"
            
        path_lightcurve = os.path.join(path_data, name)
                
        if data_format is data_formats[0]:
            if subfolder:
                os.mkdir(path_lightcurve)    
                path_lightcurve = os.path.join(path_lightcurve, name)
        if compress:
                np.savez(path_lightcurve, time=time, flux=flux)
        else:
            data = np.array([time, flux])
            np.save(path_lightcurve, data)
