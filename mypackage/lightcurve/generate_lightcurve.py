import numpy as np
from typing import Tuple

#Default value

def generate_light_curve(period:float=None, duration:float=None, epoch:float=None, observation_time:float = None, cadence:float=None, transit_depth_fraction:float=None, sigma:float=None, variation:tuple=None) -> Tuple[list, list]:
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
        observation_time = 4*300
    if cadence is None:
        cadence = 29.424/(60*24)
    if duration is None:
        duration = np.random.uniform(0.05, 0.1)
    if epoch is None:
        epoch = np.random.uniform(0, period)
    if transit_depth_fraction is None:
        transit_depth_fraction = 0.995

    time = np.arange(0, observation_time, cadence)
    flux = np.ones(time.shape[0])
    
    transits_time = np.arange(epoch, observation_time, period)

    if variation is False:
        variation = (0, 1, 0)
        left_points = np.searchsorted(time, transits_time, "left")
        right_points = np.searchsorted(time, transits_time+duration, "right")
        
    elif variation is None:
        variation_amplitude = np.random.uniform(0, duration)
        variation_frequency = np.random.uniform(1, 10)*observation_time
        variation_phase = np.random.uniform(0, 2*np.pi)
        variation = (variation_amplitude, variation_frequency, variation_phase)
    
    try:
        variation_amplitude, variation_frequency, variation_phase = variation
        time_variation = variation_amplitude*np.sin(2*np.pi*transits_time/variation_frequency + variation_phase)
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