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
    
    lc_raw_file = lc_data_path.split("/")[-1].split(".npy")[0]
    
    lc_info = csv_to_dict(cum_table, lc_raw_file)

     
    lc_info["raw_transits_time"] =  raw_transits_time.tolist()
    lc_info["raw_transits_time_no_gap"] =  raw_transits_time_no_gap.tolist()


    return lc_info
    

            


