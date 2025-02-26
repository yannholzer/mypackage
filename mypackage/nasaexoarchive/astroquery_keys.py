from enum import Enum

class TablesList(Enum):
    CUMULATIVE                              = 'cumulative'
    PLANETARY_SYSTEMS                       = 'ps'
    PLANETARY_SYSTEMS_COMPOSITE_PARAMETERS  = 'pscomppars'
    MICRO_LENSING                           = 'ml'
    ATMOSPHERIC_SPECTRA                     = 'spectra'
    TRANSITING_PLANETS                      = 'TD'
    TESS_CANDIDATES                         = 'toi'
    STELLAR_HOSTS                           = 'stellarhosts'
    KEPLER_CONFIRMED_PLANETS                = 'keplernames'
    K2_CONFIRMED_NAMES                      = 'k2names'
    KEPLER_TIMESERIES                       = 'keplertimeseries'
    K2_STELLAR_TARGETS                      = 'K2TARGETS'
    KEPLER_STELLAR                          = 'KEPLERSTELLAR'
    #                                       = "kelttimeseries",
    #                                       = 'superwasptimeseries', 
    #                                       = 'DI_STARS_EXEP', 
    #                                       = 'transitspec', 
    #                                       = 'emissionspec', 
    #                                       = 'Q1_Q6_KOI', 
    #                                       = 'Q1_Q8_KOI', 
    #                                       = 'Q1_Q12_KOI', 
    #                                       = 'Q1_Q16_KOI', 
    #                                       = 'Q1_Q17_DR24_KOI', 
    #                                       = 'Q1_Q17_DR25_KOI', 
    #                                       = 'Q1_Q17_DR25_SUP_KOI', 
    #                                       = 'Q1_Q12_TCE', 
    #                                       = 'Q1_Q16_TCE', 
    #                                       = 'Q1_Q17_DR24_TCE', 
    #                                       = 'Q1_Q17_DR25_TCE', 
    #                                       = 'stellarhosts', 
    #                                       = 'ukirttimeseries', 
    #                                       = 'object_aliases', 
    #                                       = 'k2pandc', 
    #                                       = 'Q1_Q12_KS', 
    #                                       = 'Q1_Q16_KS', 
    #                                       = 'Q1_Q17_DR24_KS', 
    #                                       = 'Q1_Q17_DR25_KS', 
    #                                       = 'Q1_Q17_DR25_SUP_KS'
    
    
class CumulativeColumns(Enum):
    
    KEPLER_ID                               = 'kepid'
    KOI_NAME                                = 'kepoi_name'
    KEPLER_NAME                             = 'kepler_name'
    DISPOSITION                             = 'koi_disposition'
    PERIOD_DAYS                             = 'koi_period'
    TIME0                                   = 'koi_time0'
    TIME0BK                                 = 'koi_time0bk'
    ECCENTRICITY                            = 'koi_eccen'
    DURATION_H                              = 'koi_duration'
    DEPTH                                   = 'koi_depth'
    INGRESS_DURATION_H                      = 'koi_ingress'
    IMPACT_PARAMETER                        = 'koi_impact'
    LONGITUDE_OF_PERIASTRON                 = 'koi_longp'
    PLANET_RADIUS_R_EARTH                   = 'koi_prad'
    SEMI_MAJOR_AXIS_AU                      = 'koi_sma'
    INCLINATION_DEG                         = 'koi_incl'
    EQUILIBRIUM_TEMPERATURE_K               = 'koi_teq'
    INSOLATION_FLUX                         = 'koi_insol'
    PLANET_STAR_RADIUS_RATIO                = 'koi_ror'
    STELLAR_DENSITY_G_CM3                   = 'koi_srho'
    MODEL_SNR                               = 'koi_model_snr'
    STELLAR_EFFECTIVE_TEMPERATURE_K         = 'koi_steff'
    STELLAR_SURFACE_GRAVITY_LOG_G           = 'koi_slogg'
    STELLAR_RADIUS_R_SUN                    = 'koi_srad'
    STELLAR_MASS_M_SUN                      = 'koi_smass'
    STELLAR_METALLICITY                     = 'koi_smet'
    STELLAR_AGE_GYR                         = 'koi_sage'
    VETTING_STATUS                          = 'koi_vet_stat'
    VETTING_DATE                            = 'koi_vet_date'
    NUMBER_OF_PLANETS                       = 'koi_count'
    NUMBER_OF_TRANSITS                      = 'koi_num_transits'
    
    
    
def usage():
    print('from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive')
    print('table = TABLE')
    print('select = COLUMN1, COLUMN2, COLUMN3, ...')
    print('where = COLUMN1 = VALUE1, COLUMN2 = VALUE2, COLUMN3 like VALUE3, COLUMN4 > VALUE4, ...')
    print("result = NasaExoplanetArchive.query_criteria(table=table, select=select, where=where)")