import csv
import numpy as np
import pandas as pd




def csv_to_dict(cumtable:str, lc_data_path:str)->dict:
    """Function to read the cumulative table and return useful information on the given light curve at the given period

    Parameters
    ----------
    cumtable : str
        the cumulative table path  
    cumtable_header: int
        The line number of the end of the header
    lc_name : str
        the light curve name
        
    Returns
    -------
    dict
        A dictionary containing the information of the lightcurve
    """
    header_line = 0
    with open(cumtable) as csvfile:
        reading = csv.reader(csvfile)
        for i_w, row in enumerate(reading):
            if not "#" in row[0]:
                header_line = i_w
                break
    
    cum_table = pd.read_csv(cumtable, sep=",", header=header_line)
    
    lc_name = lc_data_path.split("/")[-1].split(".npy")[0]
    
    lc_time, _ = np.load(lc_data_path)      
        
    lc_id = int(lc_name.split("_")[0])


    snr = float(lc_name.split("_")[1].replace("p", "."))
    period_range_min = float(lc_name.split("_")[2].replace("p", "."))
    period_range_max = float(lc_name.split("_")[3].replace("p", "."))
    period_mean = (period_range_max + period_range_min)/2

            
    if not cum_table["kepid"].isin([lc_id]).any():
        raise SystemExit("light curve not found in cumulative table.")
        
    lc_info = {}
        
    cum_table_koi = cum_table[cum_table["kepid"] == lc_id]
    
    if len(cum_table_koi.koi_period.values) > 1:
        closest_period = np.abs(period_mean - cum_table_koi.koi_period.values).argmin()

    else:
        closest_period = 0
    lc_info["lc_id"] = lc_id
    lc_info["lc_file_name"] = lc_name
    lc_info["period"] = cum_table_koi.koi_period.values[closest_period]
    lc_info["epoch"] = cum_table_koi.koi_time0bk.values[closest_period]
    lc_info["snr_file"] = snr
    lc_info["snr_table"] = cum_table_koi.koi_model_snr.values[closest_period]
    
    
    epoch = lc_info["epoch"] % lc_info["period"]
    
        
    # check if the epoch is within the first period. If yes, we start at the period number given by t0/period. Else we start at the next period number
    if (lc_time[0] % lc_info["period"]) <= epoch:
        n_period_at_t0 = np.floor(lc_time[0]/lc_info["period"]).astype(int)
    else:
        n_period_at_t0 = np.floor(lc_time[0]/lc_info["period"]).astype(int) + 1 
        
        
    # do the same for the final period
    if (lc_time[-1] % lc_info["period"]) >= epoch:
        n_period_at_tf = np.floor(lc_time[-1]/lc_info["period"]).astype(int)
    else:
        n_period_at_tf = np.floor(lc_time[-1]/lc_info["period"]).astype(int) - 1 
        
        
    start_transits_time = n_period_at_t0 * lc_info["period"] + epoch
    end_transits_time = n_period_at_tf * lc_info["period"] + epoch
    
    raw_transits_time = np.arange(start_transits_time, end_transits_time + lc_info["period"]/2, lc_info["period"])


    right = np.searchsorted(lc_time, raw_transits_time, side="right")
    #right = np.searchsorted(time, raw_transits_time, side="right")
    time_diff = np.diff(lc_time)
    median_time_diff = np.median(time_diff)
        
    raw_time_to_remove = []
    raw_transits_time_no_gap = np.copy(raw_transits_time)
    for i_t, t in enumerate(raw_transits_time):
        #print(lr_time)
        if lc_time[right[i_t]] - lc_time[right[i_t]-1] > 5*median_time_diff:
            if np.abs(t - lc_time[right[i_t]-1]) > median_time_diff and np.abs(t - lc_time[right[i_t]]) > median_time_diff:
                raw_time_to_remove.append(i_t)
        
    raw_time_to_remove = np.array(raw_time_to_remove)
    
    if raw_time_to_remove.size > 0:        
        raw_transits_time_no_gap = np.delete(raw_transits_time, raw_time_to_remove)
    
        

    
    
    lc_info["raw_transits_time"] =  raw_transits_time.tolist()
    lc_info["raw_transits_time_no_gap"] =  raw_transits_time_no_gap.tolist()


    return lc_info
    

            


