import numpy as np
from typing import Tuple
import os
import yaml


# CONSTANT #############
MIN_TO_DAYS = 1/(60*24)
############3###########

# DEFAULT LIGHT CURVE GENERATOR VALUE RANGE ##############################################
PERIOD_RANGE = (5, 10) #Days
TRANSIT_DURATION = (0.05, 0.1) #Days
OBSERVATION_TIME = 4*365 # 4 years of observation, like kepler
CADENCE = 29.4 * MIN_TO_DAYS # kepler cadence
TRANSIT_DEPTH_FRACTION = 0.995 # The fractionnal depth of the transit relative to the flux
##########################################################################################





def generate_light_curve_bkp(period:float=None, duration:float=None, epoch:float=None, observation_time:float = None, cadence:float=None, transit_depth_fraction:float=None, sigma:float=None, variation:Tuple[float, float, float]=None) -> Tuple[list, list]:
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


def generate_light_curve(period:float=None, duration:float=None, epoch:float=None, observation_time:float = None, cadence:float=None, transit_depth_fraction:float=None, sigma_noise:float=None, snr:float=None, variation:Tuple[float, float, float]=None) -> Tuple[list, list]:
    """Synthetic light curve generator. The time arguments are in unit of Days. The SNR is defined as the SNR for a single point.
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
    sigma_noise : float, optional
        The normal standard deviation of the noise injected in the transit , if None the deviation is equal to the depth of the transit.
    snr: float, optional
        The snr of the whole light curve calculated as the square root of the number of transiting points times the depth over sigma noise
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
    if snr and sigma_noise is not None:
        raise ValueError("Please enter only one argument for either the snr or the sigma noise.")

    time = np.arange(0, observation_time, cadence)
    flux = np.ones_like(time)
    
    transits_time = np.arange(epoch, observation_time, period)

        
    if variation is None:
        variation_amplitude = np.random.uniform(0, duration)
        variation_period = np.random.uniform(5, 10)*observation_time
        variation_phase = np.random.uniform(0, 2*np.pi)
    
    elif variation is False:
        variation_amplitude, variation_period, variation_phase = (0, 1, 0)    
    
    
    variation = (variation_amplitude, variation_period, variation_phase)
    
    try:
        variation_amplitude, variation_period, variation_phase = variation
        time_variation = variation_amplitude*np.sin(2*np.pi*transits_time/variation_period + variation_phase)
        left_points = np.searchsorted(time, transits_time + time_variation - duration/2, "left")
        right_points = np.searchsorted(time, transits_time + time_variation + duration/2, "right")
    
    except Exception as e:
        raise(e,"error unpacking variation parameters")
        
    
    n_transiting_points = np.sum(right_points - left_points)
    
    for i in range(transits_time.shape[0]):
        flux[left_points[i]:right_points[i]] *= transit_depth_fraction

    if sigma_noise and snr is None:
        sigma_noise = (1- transit_depth_fraction)
        snr = np.sqrt(n_transiting_points)
        flux += np.random.normal(0, sigma_noise, time.shape)
    elif sigma_noise or snr is False:
        pass
    else:
        if sigma_noise is not None:
            snr = np.sqrt(n_transiting_points)*(1-transit_depth_fraction)/ sigma_noise
            flux += np.random.normal(0, sigma_noise, time.shape)
        elif snr is not None:
            sigma_noise = np.sqrt(n_transiting_points) * (1 - transit_depth_fraction) / snr
            flux += np.random.normal(0, sigma_noise, time.shape)

    # print("sigma noise:", sigma_noise)
    # print("snr:", snr)

            
    
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
    snr:float = None,
    variation_parameters:float=None
    ):
    data_formats = ["lightcurve", "river_diagram"]
    
    if data_format not in data_formats:
        raise ValueError(f"data format not valid. please select a format from {data_formats}")
    
    period = np.random.uniform(*period_range, n_data)
    duration = np.random.uniform(*duration_range, n_data)
    
    
    path_root = os.path.join(path_to_export, data_format, "raw")
    os.makedirs(path_root, exist_ok=True)
    path_data = os.path.join(path_root, data_name)
    os.mkdir(path_data)
    for n in range(n_data):
          
        time, flux = generate_light_curve(period[n], duration[n], epoch, observation_time, cadence, transit_depth_fraction, sigma, snr, variation_parameters)
        
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



class Lightcurve:
    """Light curve class"""
    
    def __init__(
        self, 
        t0:float=0, 
        period:float=None, 
        transit_duration:float=None, 
        epoch:float=None, 
        observation_time:float = None, 
        cadence:float=None, 
        transit_depth_fraction:float=None, 
        sigma_noise:float=None, 
        snr:float=None, 
        timing_variation_params:Tuple[float, float, float]=None
        ):
    
    ### check the default value ###
        self.t0 = t0
        if period is None:
            period = np.random.uniform(*PERIOD_RANGE)
        if observation_time is None:
            observation_time = OBSERVATION_TIME
        if cadence is None:
            cadence = CADENCE
        if transit_duration is None:
            transit_duration = np.random.uniform(*TRANSIT_DURATION)
        if epoch is None:
            epoch = np.random.uniform(period)
        elif epoch is False:
            epoch = period / 2
        if transit_depth_fraction is None:
            transit_depth_fraction = TRANSIT_DEPTH_FRACTION
        if snr is not None and sigma_noise is not None:
            raise ValueError("Please enter only one argument for either the snr or the sigma noise.")
        
        if timing_variation_params is None:
            timing_variation_amplitude = np.random.uniform(0, transit_duration)
            timing_variation_period = np.random.uniform(5, 10)*observation_time
            timing_variation_phase = np.random.uniform(0, 2*np.pi)
            timing_variation_params = [timing_variation_amplitude, timing_variation_period, timing_variation_phase]
        elif timing_variation_params is False:
            timing_variation_params = [0, 1, 0]
        
        self.period = period
        self.observation_time = observation_time
        self.cadence = cadence
        self.transit_duration = transit_duration
        self.epoch = epoch
        self.transit_depth_fraction = transit_depth_fraction
        self.timing_variation_params = timing_variation_params

        timing_variation_amplitude, timing_variation_period, timing_variation_phase = self.timing_variation_params
      
        ### instantiate time and flux arrays ###        
        self.time = np.arange(self.t0, self.observation_time + self.t0, self.cadence)
        self.flux = np.ones_like(self.time)
        
        ### get the transit time according to the parameters ###
        self.transits_time = np.arange(self.epoch, self.observation_time, self.period)
        self.transits_time += timing_variation_amplitude*np.sin(2*np.pi*self.transits_time/timing_variation_period + timing_variation_phase)

        ### find the left and right limits of each transits and apply the transit fraction
        left_points = np.searchsorted(self.time, self.transits_time - self.transit_duration/2, "left")
        right_points = np.searchsorted(self.time, self.transits_time + self.transit_duration/2, "right")
        
        n_transiting_points = np.sum(right_points - left_points)
    
        for i in range(self.transits_time.shape[0]):
            self.flux[left_points[i]:right_points[i]] *= self.transit_depth_fraction
            
        self.flux_pur = np.copy(self.flux)
            
        ### add the noise ###
        if sigma_noise is None and snr is None:
            self.sigma_noise = (1- self.transit_depth_fraction)
            self.snr = np.sqrt(n_transiting_points) # because transit depth = sigma_noise, as snr = sqrt(N) * D / sigma
            self.flux += np.random.normal(0, self.sigma_noise, self.time.shape)
        elif sigma_noise is False or snr is False:
            # do not add any noise
            self.sigma_noise = self.snr = 0
            pass
        else:
            if sigma_noise is not None:
                self.sigma_noise = sigma_noise
                self.snr = np.sqrt(n_transiting_points)*(1-self.transit_depth_fraction)/ self.sigma_noise
                self.flux += np.random.normal(0, self.sigma_noise, self.time.shape)
            elif snr is not None:
                self.snr = snr
                sigma_noise = np.sqrt(n_transiting_points) * (1 - self.transit_depth_fraction) / self.snr
                self.flux += np.random.normal(0, sigma_noise, self.time.shape)
                

    def get_time_flux(self):
        return self.time, self.flux
    
    
            
        
        
class Lightcurve_npy_generator:
    def __init__(
        self,
        n_data:int, 
        path_to_export:str,
        data_name:str,
        compress:bool=False,
        subfolder:bool=False,
        data_format:str="lightcurve",
        name_period_range_format:str=True,
        save_params_txt:bool=True,
        seed:bool=False,
        period_range:Tuple[float, float]=(5, 10), 
        transit_duration_range:Tuple[float, float]=(0.05, 0.1),
        epoch:bool=None,                                        # if None, randomly selected between 0 and period, if False, set to period / 2
        observation_time:float=4*365,
        cadence:float=29.4 * MIN_TO_DAYS,
        transit_depth_fraction:float=0.995,
        sigma:float=None,
        snr:float=None,
        timing_variation_params:list[float, float, float]=None
        
        ):
        
        data_formats = ["lightcurve", "river_diagram"]
    
        if data_format not in data_formats:
            raise ValueError(f"data format not valid. please select a format from {data_formats}")
    
    
   
        path_root = os.path.join(path_to_export, data_format, "raw")
        os.makedirs(path_root, exist_ok=True)
        path_data = os.path.join(path_root, data_name)
        os.mkdir(path_data)
        
        if not isinstance(snr, list):
            snr = [snr]
            print("here")
        
        self.snr = snr
        
            
        for snr in self.snr:          
            if seed:
                np.random.seed(18111996)
            for n in range(n_data):
                period = np.random.uniform(*period_range)
                transit_duration = np.random.uniform(*transit_duration_range)
                lc = Lightcurve(0, period, transit_duration, epoch, observation_time, cadence, transit_depth_fraction, sigma, snr, timing_variation_params)
                time, flux = lc.get_time_flux()
                
            
                if name_period_range_format:
                    char = "_"
                else:
                    char = "-"
                str_snr = f"{lc.snr:.2f}".replace(".", "p")
                str_p_min = f"{period_range[0]:.2f}".replace(".", "p")
                str_p_max = f"{period_range[1]:.2f}".replace(".", "p")
                name = f"lc{str(n).zfill(len(str(n_data-1)))}_snr{str_snr}_{str_p_min}{char}{str_p_max}"
                    
                        
                
            
                
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
                
                if save_params_txt:
                   d = lc.__dict__
                   del d["time"]
                   del d["flux"]
                   del d["flux_pur"]
                   d["transits_time"] = d["transits_time"].tolist()
                   
                       
                with open(f"{path_lightcurve}.yml", "w") as f:
                    yaml.safe_dump(d, f)
                        
                

                
        
        
    

        

 