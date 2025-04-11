import xarray as xr
import netCDF4
import numpy as np
import sys

#from rhwhitepackages3.waveguides import *
from rhwhitepackages3.waveguides_pre import *
from datetime import date
import warnings

args = (sys.argv)
if len(args) < 3:
    sys.error('need at least 2 arguments: pressure level and hemisphere')
else:
    plev = int(args[1])
    hemi = args[2]


dirin = '/scratch/st-rwhite01-1/rwhite01/Waveguides_output/ERA5/'
datain = '/scratch/st-rwhite01-1/data_shared/' #'/arc/project/st-rwhite01-1/rwhite01/'

## Read in raw U data
#Ufile = '/scratch/st-rwhite01-1/data_shared/ERA5/u/ERA5_u_500_dailymean_1degree_1979-2022.nc'
Ufile = '/scratch/st-rwhite01-1/data_shared/ERA5/u/ERA5_u_850_250_dailymean_1degree_1979-2022.nc'
dirout = '/scratch/st-rwhite01-1/rwhite01/Waveguides_output/ERA5/Ks/'

startyear = 1979
endyear = 2022

if hemi == 'NH':
    latstart = 10
    latend = 90
elif hemi == 'SH':
    latstart = -90
    latend = -10
else:
    sys.error('hemisphere arguement must be "NH" or "SH"')

# Filter U data - Northern hemisphere
with xr.open_dataset(Ufile).sel(level=plev) as datain:
    # For this version of ERA5, flip latitudes
    
    try:
        if datain.latitude[0] > datain.latitude[1]:
            print('flipping latitudes')
            datain = datain.sel(latitude=slice(None, None,-1))
        # resample to daily data in case input is not daily - timefilter assumes daily data
        Udatain = datain.sel(time=slice('01-Dec-' + str(startyear),
                              '01-Jan-' + str(endyear +1))).sel(latitude=slice(latstart,latend)).u.resample(time='D').mean()
    except AttributeError:
        if datain.lat[0] > datain.lat[1]:
            print('flipping latitudes')
            datain = datain.sel(lat=slice(None, None,-1))
        # resample to daily data in case input is not daily - timefilter assumes daily data
        Udatain = datain.sel(time=slice('01-Dec-' + str(startyear),
                              '01-Jan-' + str(endyear +1))).sel(lat=slice(latstart,latend)).u.resample(time='D').mean()
    print('reached here OK')

    for wavelfilter in [2,3]:
        print(wavelfilter)
        ## Zonal filter data

        ERA5_filter_temp = zonal_filter_wind(Udatain,wavelfilter)

        for timefilter in [10,15]:
            print(timefilter)

            if timefilter == 0:
                ERA5_filter_temp2 = ERA5_filter_temp

            else:
                ERA5_filter_temp2 = butter_time_filter_wind(ERA5_filter_temp.u,timefilter)

            print('calculated')
            # Convert to datasets and append

            ds = ERA5_filter_temp2.u.to_dataset(name = 'U_filtered')

            # save file
            ds.attrs['history'] = ('Created on ' + str(date.today()) + ' by Waveguide_preprocess_ERA5.py script')
            ds.attrs['input data'] = ('Input files = ' + Ufile)
            ds.attrs['parameters'] = ('timefilter: ' + str(timefilter) + 
                                      ' and zonal wavenumber filter: ' + str(wavelfilter))

            ds.to_netcdf(dirout + '/ERA5_' + hemi + '_u' + str(plev) + '_dailymean_filtered_' + str(startyear) + '-' + str(endyear) + 
                         '_tf' + str(timefilter) + '_zfnt' + str(wavelfilter) + '_1x1.nc')

            print('written') 

# Calculate Ks

dirout = '/scratch/st-rwhite01-1/rwhite01/Waveguides_output/ERA5/Ks/'

for wavelfilter in [2,3]:     
    print(wavelfilter)

    for timefilter in [10,15]:
        print(timefilter)

        filterfilename = ('ERA5_' + hemi + '_u' + str(plev) + '_dailymean_filtered_' + str(startyear) + '-' + str(endyear) + 
		 '_tf' + str(timefilter) + '_zfnt' + str(wavelfilter) + '_1x1')

        with xr.open_dataset(dirout + '/' + filterfilename + '.nc') as ERA5_filter_temp:
            (U_temp, Ks_temp, Ks2_temp) = calc_Ks_wg(ERA5_filter_temp.U_filtered)

            print('calculated')
            # Convert to datasets and append
            ds = Ks_temp.to_dataset(name = 'Ks')
            ds['Ks2'] = Ks2_temp
            ds['Ufilter'] = U_temp

            # save file
            ds.attrs['history'] = ('Created on ' + str(date.today()) + ' by Waveguide_preprocess_ERA5.py script')
            ds.attrs['input data'] = filterfilename + ' from ' + (ERA5_filter_temp.attrs['input data'])
            ds.attrs['parameters'] = ('Timefilter: ' + str(timefilter) +' and zonal wavenumber filter: ' + str(wavelfilter))

            ds.to_netcdf(dirout + '/Ks_' + filterfilename + '.nc')

            print('written') 

