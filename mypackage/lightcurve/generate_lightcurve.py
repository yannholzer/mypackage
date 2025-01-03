import numpy as np
from typing import Tuple
import os
import yaml
import shutil
from tqdm import tqdm


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



class Lightcurve:
    """Light curve generator class

    """
    
    def __init__(
        self, 
        t0:float=None, 
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
        """_summary_

        Parameters
        ----------
        t0 : float, optional
            time of the first observation, by default is set to 0
        period : float, optional
            period of the transit contained in the light curve, in days, by default is randomly selected between PERIOD_RANGE
        transit_duration : float, optional
            duration of the transit, by default randomly selected between TRANSIT_DURATION
        epoch : float, optional
            epoch of the transit within the period, by default randomly set in the period
        observation_time : float, optional
            the total observation time, with tf = obs time + t0, by default is set to OBSERVATION_TIME
        cadence : float, optional
            the cadence of the time serie, by default is set to kepler cadence, CADENCE
        transit_depth_fraction : float, optional
            the fraction of the depth inside the transit, by default is set to TRANSIT_DEPTH_FRACTION
        sigma_noise : float, optional
            the level of the white noise, by default is the same as the depth, but overwrited by the snr if given.
        snr : float, optional
            the snr of the light curve, by default is set randomly
        timing_variation_params : Tuple[float, float, float], optional
            the TTV parameter, being [amplitude, period, phase], by default is randomly set.

        Returns
        -------
       a light curve object

        Raises
        ------
        ValueError
            if both snr and sigma are given, return an error.
        """
    
    ### check the default value ###
        if t0 is None:
            t0 = 0
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
        else:
            pass
        
        if timing_variation_params is None:
            timing_variation_amplitude = np.random.uniform(0, 1.5)*transit_duration
            timing_variation_period = np.random.uniform(0.3, 2)*observation_time
            timing_variation_phase = np.random.uniform(0, 2*np.pi)
            timing_variation_params = [timing_variation_amplitude, timing_variation_period, timing_variation_phase]
        elif timing_variation_params is False:
            timing_variation_params = [0, 1, 0]
        else: 
            #set amplitude (0) and period (0) in the range of their respective scale time
            timing_variation_params[0] *= transit_duration
            timing_variation_params[1] *= observation_time
            
        self.t0 = t0
        self.period = period
        self.observation_time = observation_time + t0
        self.cadence = cadence
        self.transit_duration = transit_duration
        self.epoch = epoch
        self.transit_depth_fraction = transit_depth_fraction
        self.timing_variation_params = timing_variation_params

        timing_variation_amplitude, timing_variation_period, timing_variation_phase = self.timing_variation_params
      
        ### instantiate time and flux arrays ###        
        self.time = np.arange(self.t0, self.observation_time, self.cadence, dtype=float)
        self.flux = np.ones_like(self.time)
        
        ### get the transit time according to the parameters ###
        self.transits_time = np.arange(self.t0 + self.epoch, self.observation_time, self.period, dtype=float)
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
            self.snr = np.random.uniform(6, 25)
            #as snr = sqrt(N) * D / sigma
            self.sigma_noise = np.sqrt(n_transiting_points) * (1 - self.transit_depth_fraction) / self.snr
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
                self.sigma_noise = np.sqrt(n_transiting_points) * (1 - self.transit_depth_fraction) / self.snr
                self.flux += np.random.normal(0, self.sigma_noise, self.time.shape)
                

    def get_time_flux(self):
        return self.time, self.flux
    
    def get_time_flux_and_pur(self):
        return self.time, self.flux, self.flux_pur
    
    
    def median_removed_normalized(self):
        self.flux -= np.median(self.flux)
        self.flux/=np.linalg.norm(self.flux)
        return self
    
            
        
        
class Lightcurve_npy_generator_snr_range_seedable:
    def __init__(
        self,
        n_data:int, 
        path_to_export:str,
        data_name:str,
        compress:bool=False,
        subfolder:bool=False,
        data_format:str="synthetic",
        name_period_range_format:str=True,
        save_params_txt:bool=True,
        seed:bool=False,
        period_range:Tuple[float, float]=(5, 10), 
        transit_duration_range:Tuple[float, float]=(0.05, 0.1),
        epoch:bool=None,                                        # if None, randomly selected between 0 and period, if False, set to period / 2
        t0:float=0,
        observation_time:float=4*365,
        cadence:float=29.4 * MIN_TO_DAYS,
        transit_depth_fraction:float=0.995,
        sigma:float=None,
        snr:float=None,
        timing_variation_params:list[float, float, float]=None
        
        ):
        
        data_formats = ["synthetic", "river_diagram"]
    
        if data_format not in data_formats:
            raise ValueError(f"data format {data_format} not valid. please select a format from {data_formats}")
    
    
   
        path_root = os.path.join(path_to_export, data_format, "raw")
        os.makedirs(path_root, exist_ok=True)
        path_data = os.path.join(path_root, data_name)
        try:
            os.mkdir(path_data)
        except Exception as e:
            print(e)
            try:
                entry = input("do you want to erase current datas ? (y/N)")
            except EOFError:
                print("script is running remotely, delete file manually.")
                entry = False
            if entry != "y":
                raise(SystemExit)
            shutil.rmtree(path_data)
            os.mkdir(path_data)
                
        
        if not isinstance(snr, list):
            snr = [snr]
        
        self.snr = snr
        
            
        for snr in self.snr:          
            if seed is not None:
                np.random.seed(seed)
            for n in range(n_data):
                period = np.random.uniform(*period_range)
                transit_duration = np.random.uniform(*transit_duration_range)
                lc = Lightcurve(t0, period, transit_duration, epoch, observation_time, cadence, transit_depth_fraction, sigma, snr, timing_variation_params)
                time, flux = lc.get_time_flux()
                
            
                if name_period_range_format:
                    char = "_"
                else:
                    char = "-"
                str_snr = f"{lc.snr:.2f}".replace(".", "p")
                str_p_min = f"{period_range[0]:.2f}".replace(".", "p")
                str_p_max = f"{period_range[1]:.2f}".replace(".", "p")
                name = f"lc{str(n).zfill(len(str(n_data-1)))}_{str_snr}_{str_p_min}{char}{str_p_max}"
                    
                        
                
            
                
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
                   d["snr"] = float(d["snr"])
                   d["sigma_noise"] = float(d["sigma_noise"])
                   
                   del d["time"]
                   del d["flux"]
                   del d["flux_pur"]
                   d["transits_time"] = d["transits_time"].tolist()
                   
                       
                with open(f"{path_lightcurve}.yml", "w") as f:
                    yaml.safe_dump(d, f)
                        
                

class Lightcurve_npy_generator_argument_range:
    def __init__(
        self,
        n_data:int, 
        path_to_export:str,
        data_name:str,
        compress:bool=False,
        subfolder:bool=False,
        data_format:str="synthetic",
        name_period_range_format:str=True,
        save_params_txt:bool=True,
        init_period_range:Tuple[float, float]=(5, 10), 
        window_range:Tuple[float, float]= (0.2, 0.8),
        transit_duration_range:Tuple[float, float]=(0.05, 0.1),
        epoch:bool=None,                                        # if None, randomly selected between 0 and period, if False, set to period / 2
        t0_range:Tuple[float, float]=(0, 0),
        observation_time:float=4*365,
        cadence:float=29.4 * MIN_TO_DAYS,
        transit_depth_fraction:float=0.995,
        snr_range:Tuple[float, float]=(2, 25),
        sigma_noise:bool=None,
        timing_variation_params_range:list[float, float, float]=[(1, 5), (0.3, 2), (0, 2*np.pi)]
        ):
        
        data_formats = ["synthetic", "river_diagram"]
    
        if data_format not in data_formats:
            raise ValueError(f"data format not valid. please select a format from {data_formats}")
    
    
   
        path_root = os.path.join(path_to_export, data_format, "raw")
        os.makedirs(path_root, exist_ok=True)
        path_data = os.path.join(path_root, data_name)
        try:
            os.mkdir(path_data)
        except Exception as e:
            print(e)
            try:
                entry = input("do you want to erase current datas ? (y/N)")
            except EOFError:
                print("script is running remotely, delete file manually.")
                entry = False
            if entry != "y":
                raise(SystemExit)
            shutil.rmtree(path_data)
            os.mkdir(path_data)
                
        
    
        
        for n in tqdm(range(n_data), desc="creating data"):
            period = np.random.uniform(*init_period_range)
            window = np.random.uniform(*window_range)
            period_range = (period-window/2, period+window/2)
            period = np.random.uniform(*period_range)
            transit_duration = np.random.uniform(*transit_duration_range)
            if not snr_range:
                if not sigma_noise:
                    print("both sigma and snr null, exit.")
                    raise SystemExit
                snr = None
            else:
                snr = np.random.uniform(*snr_range)
            t0 = np.random.uniform(*t0_range)
            if not timing_variation_params_range:
                timing_variation_params = False
            else:
                timing_variation_params = [
                    np.random.uniform(*timing_variation_params_range[0]),
                    np.random.uniform(*timing_variation_params_range[1]),
                    np.random.uniform(*timing_variation_params_range[2]),
            ]
            
            lc = Lightcurve(t0, period, transit_duration, epoch, observation_time, cadence, transit_depth_fraction, sigma_noise, snr, timing_variation_params)
            time, flux = lc.get_time_flux()
            
            str_snr = f"{lc.snr:.2f}".replace(".", "p")
            if name_period_range_format:
                str_p_min = f"{period_range[0]:.2f}".replace(".", "p")
                str_p_max = f"{period_range[1]:.2f}".replace(".", "p")
                name = f"lc{str(n).zfill(len(str(n_data-1)))}_{str_snr}_{str_p_min}_{str_p_max}"
            else:
                name = f"lc{str(n).zfill(len(str(n_data-1)))}_{str_snr}"
                
                    
            
        
            
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
                d["snr"] = float(d["snr"])
                d["sigma_noise"] = float(d["sigma_noise"])
                
                del d["time"]
                del d["flux"]
                del d["flux_pur"]
                d["transits_time"] = d["transits_time"].tolist()
                
                    
            with open(f"{path_lightcurve}.yml", "w") as f:
                yaml.safe_dump(d, f)         
    
    



        
        
        
    
    
    

        

 