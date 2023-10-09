from bin_lightcurve import bin_lightcurve

def create_river_diagram(time, flux, period):
    new_time, new_flux, new_cadence, river_diagram_shape = bin_lightcurve(time, flux, period=period)
    
    river_diagram = flux.reshape(river_diagram_shape)
    
    return river_diagram
    
    