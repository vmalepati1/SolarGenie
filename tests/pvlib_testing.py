import pvlib
from topology_estimation.get_elevation import elevation_function
import pandas as pd

latitude = 33.9715556589973
longitude = -84.4506138363946
altitude = elevation_function((latitude, longitude))

times = pd.date_range('06/21/2018 14:00', periods=1, freq='20min')
times = times.tz_localize('EST')


solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
dni_extra = pvlib.irradiance.get_extra_radiation(times)
airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
pressure = pvlib.atmosphere.alt2pres(altitude)
am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
                             dni_extra=dni_extra, altitude=altitude)

print(cs['dni'])
print(solpos['azimuth'].item())
print(90 - solpos['apparent_zenith'].item())
# total_irrad = pvlib.irradiance.get_total_irradiance(system['surface_tilt'],
#                                                     system['surface_azimuth'],
#                                                     solpos['apparent_zenith'],
#                                                     solpos['azimuth'],
#                                                     cs['dni'], cs['ghi'], cs['dhi'],
#                                                     dni_extra=dni_extra,
#                                                     model='haydavies')