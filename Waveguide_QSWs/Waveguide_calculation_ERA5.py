import xarray as xr
import netCDF4
import numpy as np
import sys

from rhwhitepackages3.CESMconst import *
from rhwhitepackages3.waveguides import *

from datetime import date
import warnings

args = (sys.argv)
if len(args) < 5:
    sys.error('need at least 4 arguments: wavelength, timefilter, plev and hemisphere')
else:
    wavelfilter = args[1]
    timefilter = args[2]
    plev = args[3]
    hemi = args[4]


dirin = '/scratch/st-rwhite01-1/rwhite01/Waveguides_output/ERA5/'
datain = '/scratch/st-rwhite01-1/data_shared/' #'/arc/project/st-rwhite01-1/rwhite01/'

# Set variable thresholds
Uthresh = 0.5 # minimum windspeed of 0.5m/s to count as part of the waveguide (must be >0)
minwidth = 5 # minimum width of waveguide in degrees latitude. 5 would be one node of a wave 
             # with meridional wavelength of 10 degrees, however many narrow jets don't produce
             # waveguides much wider than this. 
             # Worth testing the sensitivity to this. 
mindepth = 1 # minimum depth to count as a waveguide in units of wavenumber (Ks)
wnstart = 4 # Ks value to start looking for waveguides at
wnend = 9 # maximum Ks value to look for waveguides from

Ufile = datain + 'ERA5/u/ERA5_u_500_dailymean_1degree_1979-2022.nc'

startdate = '01-Jan-1979'
enddate = '01-Jan-2023'

# Northern hemisphere

if hemi == 'NH':
    latstart=20
    latend = 85
elif hemi == 'SH':
    latstart=-85
    latend = -20
else:
    sys.error('hemisphere argument must be "NH" or "SH"')


with xr.open_dataset(dirin +
                    'Ks/Ks_ERA5_' + hemi + '_u' + str(plev) + '_dailymean_filtered_1979-2022_tf' + str(timefilter) +
                    '_zfnt' + str(wavelfilter) + '_1x1.nc') as filein:

    U_datain = filein.sel(time=slice(startdate,enddate)).Ufilter.load()
    Ks_datain = filein.sel(time=slice(startdate,enddate)).Ks.load()
    Ks2_datain = filein.sel(time=slice(startdate,enddate)).Ks2.load()
    print('loaded Ks')  

    wgs,wgs_map = calc_waveguide_map(U_datain,Ks_datain,Ks2_datain,latstart,latend,
                                      Uthresh,wnstart,wnend,minwidth,mindepth,Ufile,interp=True,toprint=False)

    dirout = '/scratch/st-rwhite01-1/rwhite01/Waveguides_output/ERA5/Waveguides/'

    wgs.to_netcdf(dirout + '/ERA5_' + hemi + '_' + str(plev) + 'mb_dailymean_1x1_1979-2022_WGs_' +
                          'tf' + str(timefilter) + '_zfnt' + str(wavelfilter) + '_' + 
                          str(Uthresh) +   '_' + str(mindepth) + '_' + str(minwidth) + '.nc')

    WGmap_file = ('ERA5_' + hemi + '_' + str(plev) + 'mb_dailymean_1x1_1979-2022_WGmap_' +
                          'tf' + str(timefilter) + '_zfnt' + str(wavelfilter) + '_' +
                          str(Uthresh) +   '_' + str(mindepth) + '_' + str(minwidth))
    wgs_map.to_netcdf(dirout + '/' + WGmap_file + '.nc')

print('complete')

# Calculate climatologies
calc_waveguide_clims(hemi,plev,dirout,WGmap_file)

print('climatologies complete')
