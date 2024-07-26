import numpy as np
from typing import Tuple, NamedTuple
from dataclasses import dataclass, asdict
from collections import namedtuple
from mypackage.lightcurve.bin_lightcurve import bin_lightcurve_faster
from bls.src.algo.bls import slide_window_sr





def get_snr_from_bls(time: np.ndarray, 
                     flux: np.ndarray, 
                     transit_time: np.ndarray, 
                     duration_range: Tuple[float, float] = None, 
                     bls_binning_cadence: float = 10/60/24, 
                     time_window: float = 600/60/24, 
                     transit_parameters: bool = False,
                     stacked_time_flux: bool = False
                     ) -> float | Tuple[float, NamedTuple]:
    """Calculate the snr and the transit parameters with the BLS method for a given array of transit time.

    Parameters
    ----------
    time : np.ndarray
        The time array of the light curve.
    flux : np.ndarray
        The flux array of the light curve.
    transit_time : np.ndarray
        The array of transit time.
    duration_range : Tuple[float, float], optional
        The duration range of the transit the bls looks for, by default None and set between 30 min and twice the size of the time window.
    bls_binning_cadence : float, optional
        The binning cadence of the bls algorithm, by default 10 mins.
    time_window : float, optional
        The time window to capture around the transit time , by default 600 mins.
    transit_parameters : float, optional
        If True, also return the transit depth and duration.

    Returns
    -------
    out: float
        The snr, the transit depth and the transit duration if transit_parameters is set to True.
    
    transit_paramets: NamedTuple[float, float], optional
        depth: float
            The transit depth.
        duration: float
            The transit duration.
    stacked_time_flux: NamedTuple[np.ndarray, np.ndarray], optional
        stacked_time: np.ndarray
        The stacked time array gathered from each given transit time.
        stacked_flux: np.ndarray
        The stacked flux array gathered from each given transit time.
    """
    # set the default duration range between 30 min and twice the size of the time window
    if duration_range is None:
        duration_range = (30/60/24, 2*time_window)
    
    flux_error = np.ones_like(flux) * flux.std()
    with np.errstate(divide='ignore'):
        weights = flux_error**(-2) * (1 / np.sum(flux_error**(-2)))
    
    left = np.searchsorted(time, transit_time - time_window, side="right")
    right = np.searchsorted(time, transit_time + time_window, side="left")
                
    gather_time = []
    gather_flux = []
    gather_weights = []
    for i, tt in enumerate(transit_time):
        gather_time += (time[left[i]:right[i]] - tt).tolist()
        gather_flux += flux[left[i]:right[i]].tolist()
        gather_weights += weights[left[i]:right[i]].tolist()
        
    gather_time = np.array(gather_time)
    gather_flux = np.array(gather_flux)
    gather_weights = np.array(gather_weights)
    
    sort_arg = np.argsort(gather_time)
    gather_time = gather_time[sort_arg]
    gather_flux = gather_flux[sort_arg]
    gather_weights = gather_weights[sort_arg]

    binned_time, binned_flux, _ = bin_lightcurve_faster(gather_time, gather_flux, cadence=bls_binning_cadence, fill_between=0)
    _, binned_weights, _ = bin_lightcurve_faster(gather_time, gather_weights, cadence=bls_binning_cadence, fill_between=0)
    
    # substract the flux to 1 to get normalized the flux at zeros
    best_sr, best_t, best_s_r = slide_window_sr((duration_range), binned_time, 1-binned_flux, binned_weights)
    s, r = best_s_r
    
    if r == 0:
        duration = 0
        depth = 0
        snr = 0
    else:
        depth  = best_sr / np.sqrt(r * (1-r))

        start_t_index = np.abs(gather_time-binned_time[best_t[0]]).argmin()
        end_t_index = np.abs(gather_time-binned_time[best_t[1]]).argmin()
                
        end_t = gather_time[end_t_index]
        start_t = gather_time[start_t_index]
        
        duration = end_t - start_t
        left = np.searchsorted(time, transit_time - duration, side="right")
        right = np.searchsorted(time, transit_time + duration, side="left")

        in_transit_mask = np.zeros_like(time, dtype=bool)
        for i, tt in enumerate(transit_time):    
            in_transit_mask[left[i]:right[i]] = True
    
        out_transit = flux[~in_transit_mask]
        
        snr = np.sqrt(end_t_index-start_t_index) * depth / np.std(out_transit)
                        
    
    opt_return = {}
    
    if transit_parameters:
        opt_return["depth"] = depth
        opt_return["duration"] = duration
    
    if stacked_time_flux:
        opt_return["stacked_time"] = gather_time
        opt_return["stacked_flux"] = gather_flux
            
    
    if opt_return:
        nametuple_return = namedtuple("opt_return", [*opt_return])(**opt_return)
        return snr, nametuple_return
    
    
    return snr
    
    
def transit_time_to_TTV(time: np.ndarray, 
                        transit_time: np.ndarray, 
                        period: float
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Convert the transit time into TTV and transit number.

    Parameters
    ----------
    time : np.ndarray
        The time array of the light curve.
    transit_time : np.ndarray
        The array of transit time.
    period : float
        The period of the transit.

    Returns
    -------
    TTV : np.ndarray
        The TTV array.
    transit_number : np.ndarray
        The transit number array.
    """

    ttv = (transit_time-time[0]) % period
    transit_number = np.floor((transit_time-time[0]) / period)
    for i in range(ttv.size - 1):
        if ttv[i+1] > (ttv[i] + period/2):
            ttv[i+1:] -= period
        elif ttv[i+1] < (ttv[i] - period/2):
            ttv[i+1:] += period  
    
    return ttv, transit_number



    
    
def sin_fit(observation_time: float, 
            transit_time: np.ndarray, 
            period: float, 
            frequency_grid_resolution: int = 51,
            ) -> NamedTuple:
    """Fit a linear + sinusoidal model on the transit time to get the best fit parameters.

    Parameters
    ----------
    observation time : float
        The total observation time of the light curve.
    transit_time : np.ndarray
        The array of transit time.
    period : float
        The period of the transit.
    frequency_grid_resolution : int, optional
        The resolution of the frequency grid the sinusoidal fit is tested on, by default 51.

    Returns
    -------
    parameters: NamedTuple
        A named tuple containing the best fit parameters listed below:
    \n
    a : float
        The slope of the fit.
    b : float
        The intercept of the fit.
    period : float
        The best fit period
    a_sin : float
        The amplitude of the sinus parameters.
    a_cos : float
        The amplitude of the cosinus parameters.
    frequency : float
        The best fit frequency.
    reduced_chi2 : float
        The reduced chi-square of the best fit.
    """
    tt0 = transit_time[0] + period/2
    transit_time = transit_time - tt0
    methods = ["min_diff", "median_diff", "period_mean", "period_rd", "double_fit"]
    @dataclass
    class Parameters:
        period: float
        linear_coeff: float = 0
        linear_intersect: float = 0
        sin_amplitude: float = 0
        cos_amplitude: float = 0
        frequency: float = 0
        reduced_chi2: float = np.inf
        
    best_params = Parameters(period)
        
    for method in methods:
        if method == "min_diff":
            if transit_time.size < 2:
                continue
            period_fit = np.min(np.diff(transit_time))
        if method == "median_diff":
            if transit_time.size < 2:
                continue
            period_fit = np.median(np.diff(transit_time))
        if method == "period_rd":
            period_fit = period
        if method == "double_fit":
            period_fit = best_params.linear_coeff
    
        # convert the transit time into TTV and transit number
        #ttv, transit_number = transit_time_to_TTV(time, transit_time, period_fit)
        transit_number = np.floor((transit_time) / period_fit)
        
        
        min_period_freq = 10 # number of lines, translate to the number of orbital periods
        max_period_freq = 2 * (observation_time/period_fit)
        freq = np.linspace(1/max_period_freq, 1/min_period_freq, frequency_grid_resolution)   
        for f in freq:                     
            A = np.vstack([np.sin(2*np.pi*f*transit_number), np.cos(2*np.pi*f*transit_number), transit_number, np.ones_like(transit_number)]).T
            p = np.linalg.lstsq(A, transit_time, rcond=None)[0]
            sin_amplitude, cos_amplitude, linear_coeff, linear_intersect = p
            
            transit_time_model = A @ p
            
            reduced_chi2 = reduced_chi2_f(transit_time, transit_time_model)
                        
            if reduced_chi2 < best_params.reduced_chi2:
                best_params.linear_coeff = linear_coeff
                best_params.linear_intersect = linear_intersect + tt0
                best_params.sin_amplitude = sin_amplitude
                best_params.cos_amplitude = cos_amplitude
                best_params.frequency = f
                best_params.period = period_fit
                best_params.reduced_chi2 = reduced_chi2
                
            
    
    best_params = asdict(best_params)
    result = namedtuple("best_params", [*best_params])(**best_params)
    

        
    return result




def linear_fit(transit_time: np.ndarray, 
               period: float
               ) -> NamedTuple:
    """Fit a linear model on the transit time to get the best fit parameters.

    Parameters
    ----------
    transit_time : np.ndarray
        The array of transit time.
    period : float
        The period of the transit.
   
    Returns
    -------
    parameters: NamedTuple
        A named tuple containing the best fit parameters listed below:
    \n
    a : float
        The slope of the fit.
    b : float
        The intercept of the fit.
    period : float
        The best fit period
    reduced_chi2 : float
        The reduced chi-square of the best fit.
    """
    tt0 = transit_time[0] + period/2
    transit_time = transit_time - tt0
    methods = ["min_diff", "median_diff", "period_mean", "period_rd", "double_fit"]
    @dataclass
    class Parameters:
        period: float
        linear_coeff: float = 0
        linear_intersect: float = 0
        reduced_chi2: float = np.inf
        
    best_params = Parameters(period)
        
    for method in methods:
        if method == "min_diff":
            if transit_time.size < 2:
                continue
            period_fit = np.min(np.diff(transit_time))
        if method == "median_diff":
            if transit_time.size < 2:
                continue
            period_fit = np.median(np.diff(transit_time))
        if method == "period_rd":
            period_fit = period
        if method == "double_fit":
            period_fit = best_params.linear_coeff
    
        # convert the transit time into TTV and transit number
        #ttv, transit_number = transit_time_to_TTV(time, transit_time, period_fit)
        transit_number = np.floor((transit_time) / period_fit)
        
        
        
        A = np.vstack([transit_number, np.ones_like(transit_number)]).T 
        p = np.linalg.lstsq(A, transit_time, rcond=None)[0]
        linear_coeff, linear_intersect = p
        
        transit_time_model = A @ p
        
        reduced_chi2 = reduced_chi2_f(transit_time, transit_time_model)
                    
        if reduced_chi2 < best_params.reduced_chi2:
            best_params.linear_coeff = linear_coeff
            best_params.linear_intersect = linear_intersect + tt0
            best_params.period = period_fit
            best_params.reduced_chi2 = reduced_chi2
            

    
    best_params = asdict(best_params)
    result = namedtuple("best_params", [*best_params])(**best_params)
    

        
    return result




def reduced_chi2_f(transit_time: np.ndarray, transit_time_model: np.ndarray, transit_time_error: float = 30 / 60 / 24) -> float:
    """Calculate the reduced chi2 of the transit time fit.

    Parameters
    ----------
    transit_time : np.ndarray
        The transit time array.
    transit_time_model : np.ndarray
        The transit time model array.
    transit_time_error : float, optional
        The error on the transit time, by default 30 mins

    Returns
    -------
    float
        The reduced chi2.
    """
    chi2 = np.sum((transit_time - transit_time_model)**2 / transit_time_error**2)
    reduced_chi2 = chi2 / (transit_time.size)
    
    return reduced_chi2




def transit_time_from_fit(params: list, 
                          time: np.ndarray,
                          transit_time: np.ndarray = None, 
                          mode: str = "sin",
                          complet: bool = False
                          ) -> np.ndarray:
    """Get the transit time from the linear sin model.
    
    Parameters
    ----------
    params : NamedTuple
        The linear sin model parameters.
    time : np.ndarray
        The time array of the light curve.
    period : float
        The period of the transit.
    mode: str, optional
        The mode of the fit, by default "sin", can be "sin", "only_sin" or "linear".
    complet : bool, optional
        If True, return the transit time for the whole light curve, by default False

    Returns
    -------
    np.ndarray
        The transit time.
    """
    if mode == "sin":
        linear_coeff, linear_intersect, period, sin_amplitude, cos_amplitude, frequency = params
    elif mode == "only_sin":
        linear_coeff, linear_intersect, period, sin_amplitude, cos_amplitude, frequency  = params
    elif mode == "linear":
        linear_coeff, linear_intersect, period  = params[:3]
    else:
        print("Mode invalid. Choose between 'sin', 'only_sin' or 'linear'")
        raise ValueError
    
    tt0 = transit_time[0] + period/2
    if complet:
        start_number = np.floor((time[0]-transit_time[0]) / period)
        end_number = np.floor((transit_time[-1]-tt0) / period) + np.floor((time[-1]-transit_time[-1])/period)
        transit_number = np.arange(start_number, end_number)
    else:
        transit_number = np.floor((transit_time - transit_time[0]-period/2) / period)
        
        
        
    if mode == "sin":
        A = np.vstack([np.sin(2*np.pi*frequency*transit_number), np.cos(2*np.pi*frequency*transit_number), transit_number, np.ones_like(transit_number)]).T
        transit_time = A @ [sin_amplitude, cos_amplitude, linear_coeff, linear_intersect]
    elif mode == "only_sin":
        A = np.vstack([np.sin(2*np.pi*frequency*transit_number), np.cos(2*np.pi*frequency*transit_number)]).T
        transit_time = A @ [sin_amplitude, cos_amplitude]
    elif mode == "linear":
        A = np.vstack([transit_number, np.ones_like(transit_number)]).T
        transit_time = A @ [linear_coeff, linear_intersect]

        
    return transit_number, transit_time



    

def likelihood_from_transit_time(time: np.ndarray, flux: np.ndarray, transit_time: np.ndarray, duration: np.ndarray, depth: np.ndarray, log_likelihood: bool = True) -> float:
    
    flux_model = np.ones_like(flux)
    for tt in transit_time:
        left = np.searchsorted(time, tt - duration, side="right")
        right = np.searchsorted(time, tt + duration, side="left")
        flux_model[left:right] -= depth
    
    in_transit = flux_model < 1
    out_transit = flux_model >= 1
    flux_noise = np.std(flux[out_transit])
    
    if log_likelihood:
        log_likelihood = - flux.size * np.log(2*np.pi*flux_noise) - 1/2 * np.sum((flux - flux_model)**2 / (2*flux_noise**2))
        return log_likelihood
    else:
        print("not implemented yet")
        
    
        
    
    
    