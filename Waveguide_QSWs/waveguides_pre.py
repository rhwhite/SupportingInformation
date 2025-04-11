# Module to identify waveguides in refractive index data
# Written by R. H. White rwhite@eoas.ubc.ca
import numpy as np
import xarray as xr
import math
from scipy.interpolate import interp1d
from scipy import signal
from scipy.signal import butter, lfilter, sosfilt
import scipy.fftpack as fftpack

from rhwhitepackages3.physconst import *

from datetime import date

## Preprocess code - final
def lowpass_butter(data,day1,fs,order=5):
    lowcut=2.0 * (1.0/day1) * (1.0/fs) # fraction of Nyquist frequency; 2.0 because Nyquist frequency is 0.5 samples per days
    sos = butter(order, lowcut, btype='lowpass',output='sos') #low pass filter

    # run filter forwards and backwards to get zero phase shift
    filtered = signal.sosfiltfilt(sos, data, axis=0)

    try:
        xrtimefilt = xr.DataArray(filtered,coords={'time':data.time,
                                             'longitude':data.longitude.values,'latitude':data.latitude.values},
                                             dims = ('time','latitude','longitude'))

    except:
        xrtimefilt = xr.DataArray(filtered,coords={'time':data.time,
                                             'longitude':data.lon.values,'latitude':data.lat.values},
                                             dims = ('time','latitude','longitude'))
        
        
    return(xrtimefilt)

def butter_time_filter_wind(infile,cutoff,varname='u'):
    #datain_noleap = infile.sel(time=~((infile.time.dt.month == 2) & (infile.time.dt.day == 29)))

    # Get appropriate weights, convolve, and select every 5th timestep
    nwghts = 31
    fs = 1.0          # 1 per day in 1/days (sampling frequency)
    day1     = cutoff #days

    xrtimefilt = lowpass_butter(infile,day1,fs)
    xrtimefilt = xrtimefilt.to_dataset(name=varname)
   
    return(xrtimefilt)


def fourier_wind(infile,indata,nlons,peak_freq,ndegs=360):
    X_fft = fftpack.fft(indata)

    f_s = nlons
    freqs = fftpack.fftfreq(len(indata)) * f_s
    t = np.linspace(0, ndegs, f_s, endpoint=False)

    filt_fft = X_fft.copy()
    filt_fft[np.abs(freqs) > peak_freq] = 0
    filtered_sig = fftpack.ifft(filt_fft)

    return(filtered_sig,t)

def fourier_Tukey(infile,indata,nlons,peak_freq,ndegs=360):
    X_fft = fftpack.fft(indata)

    f_s = nlons
    freqs = fftpack.fftfreq(len(indata)) * f_s
    t = np.linspace(0, ndegs, f_s, endpoint=False)

    filt_fft = X_fft.copy()
    filt_fft[np.abs(freqs) > peak_freq] = 0
    filtered_sig = fftpack.ifft(filt_fft)

    # create Tukey window to smooth the wavenumbers removed (so no exact cutoff at k=2, 
    #which will change at different latitudes)
    # Window is 2 wavenumbers more than the peak, but multiplied by 2 because the Tukey window is symmetric
    M = (peak_freq + 2)*2
    alpha = 0.3 # co-sine weighting covers 30% of the window
    tukeyWin = signal.tukey(M, alpha=0.3, sym=True)[int(M/2):M]

    turfilt_fft = X_fft.copy()
    n = len(turfilt_fft)
    turfilt_fft[0:int(M/2)] = turfilt_fft[0:int(M/2)]*tukeyWin
    turfilt_fft[int(M/2):n-int(M/2)] = 0
    turfilt_fft[n-int(M/2):n] = turfilt_fft[n-int(M/2):n]*tukeyWin[::-1]
    tur_filtered_sig = fftpack.ifft(turfilt_fft)

    return(tur_filtered_sig,filtered_sig,t)

def zonal_filter_wind(infile,peak_freq):
    ntimes = len(infile.time)
    std_filt_data = np.ndarray(infile.shape)
    
    try:
        nlats = len(infile.latitude)   

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(latitude=ilat).isel(time=itime)
                std_filt_data[itime,ilat,:],t = fourier_wind(
                                                infile,x.values,len(x.longitude),peak_freq=peak_freq)

        data_stdfilt = xr.DataArray(std_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.longitude,'latitude':infile.latitude},
                                              dims = ('time','latitude','longitude'))

    except:
        nlats = len(infile.lat)

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(lat=ilat).isel(time=itime)
                std_filt_data[itime,ilat,:],t = fourier_wind(
                                                infile,x.values,len(x.lon),peak_freq=peak_freq)

        data_stdfilt = xr.DataArray(std_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.lon.values,'latitude':infile.lat.values},
                                              dims = ('time','latitude','longitude'))

    
    data_stdfilt = data_stdfilt.to_dataset(name='u')
    
    return(data_stdfilt)

def zonal_filter_PV(infile,peak_freq):
    ntimes = len(infile.time)
    std_filt_data = np.ndarray(infile.shape)

    try:
        nlats = len(infile.latitude)

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(latitude=ilat).isel(time=itime)
                std_filt_data[itime,ilat,:],t = fourier_wind(
                                                infile,x.values,len(x.longitude),peak_freq=peak_freq)

        data_stdfilt = xr.DataArray(std_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.longitude,'latitude':infile.latitude},
                                              dims = ('time','latitude','longitude'))

    except:
        nlats = len(infile.lat)

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(lat=ilat).isel(time=itime)
                std_filt_data[itime,ilat,:],t = fourier_wind(
                                                infile,x.values,len(x.lon),peak_freq=peak_freq)

        data_stdfilt = xr.DataArray(std_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.lon.values,'latitude':infile.lat.values},
                                              dims = ('time','latitude','longitude'))


    data_stdfilt = data_stdfilt.to_dataset(name='pv')

    return(data_stdfilt)


def zonal_filter_wind_tur(infile,peak_freq):
    ntimes = len(infile.time)
    tur_filt_data = np.ndarray(infile.shape)

    try:
        nlats = len(infile.latitude)

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(latitude=ilat).isel(time=itime)
                tur_filt_data[itime,ilat,:],std_filt_data[itime,ilat,:],t = fourier_Tukey(
                                                infile,x.values,len(x.longitude),peak_freq=peak_freq)

        data_turfilt = xr.DataArray(tur_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.longitude,'latitude':infile.latitude},
                                              dims = ('time','latitude','longitude'))

    except:
        nlats = len(infile.lat)

        for itime in range(0,ntimes):
            for ilat in range(0,nlats):
                x = infile.isel(lat=ilat).isel(time=itime)
                tur_filt_data[itime,ilat,:],std_filt_data[itime,ilat,:],t = fourier_Tukey(
                                                infile,x.values,len(x.lon),peak_freq=peak_freq)

        data_turfilt = xr.DataArray(tur_filt_data,coords={'time':infile.time,
                                                     'longitude':infile.lon.values,'latitude':infile.lat.values},
                                              dims = ('time','latitude','longitude'))


    data_turfilt = data_turfilt.to_dataset(name='u')


def old_calc_Ks_SG(Uin,SG_step1=0,SG_step2=0,winlen=41):
    ## Calculate BetaM
    ## Hoskins and Karoly (see also Vallis (page 551) and Petoukhov et al 2013
    ## and Hoskins and Ambrizzi (1993))

    OMEGA = 7.2921E-5
    a = 6.3781E6
    try:
        lats_r = np.deg2rad(Uin.latitude)
    except AttributeError:
        lats_r = np.deg2rad(Uin.lat)

    coslat = np.cos(lats_r)

    betaM1 = 2.0 * OMEGA * coslat * coslat / a

    Um = Uin / coslat

    cos2Um = Um * coslat * coslat

    # first differentiation
    ddy_1 = ddy_merc(cos2Um)
    # divide by cos2phi
    ddy_1_over_cos2p = ddy_1 * (1.0/(coslat * coslat))

    # Apply Savitzky-Golay filter
    if SG_step1 > 0:
        # Check that axis 1 is latitude
        if Uin.dims[1] in ['latitude','lat','lats']:
            temp = signal.savgol_filter(ddy_1_over_cos2p, 
                                            window_length=winlen, polyorder=SG_step1, 
                                            axis=1)
            
            ddy_1_over_cos2p = xr.DataArray(temp,coords={'time':ddy_1_over_cos2p.time,
                                                'longitude':ddy_1_over_cos2p.longitude,
                                                'latitude':ddy_1_over_cos2p.latitude},
                                          dims = ('time','latitude','longitude'))

        else:
            error('latitude axis is not as expected, or not named latitude, lat or lats')
    
    # second differentiation
    ddy_2 = ddy_merc(ddy_1_over_cos2p)

    # Apply Savitzky-Golay filter
    if SG_step2 > 0:
        # Check that axis 1 is latitude
        if Uin.dims[1] in ['latitude','lat','lats']:
            temp = signal.savgol_filter(ddy_2, 
                                            window_length=winlen, polyorder=SG_step2, 
                                            axis=1)
            ddy_2 = xr.DataArray(temp,coords={'time':ddy_2.time,
                                                'longitude':ddy_2.longitude,
                                                'latitude':ddy_2.latitude},
                                          dims = ('time','latitude','longitude'))

            
        else:
            error('latitude axis is not as expected, or not named latitude, lat or lats')
    
    
    betaM = betaM1 - ddy_2
    # Now calculate Ks from BetaM
    Ks2 = a * a * betaM/Um

    Ks = np.sqrt(Ks2)

    return(ddy_1,Ks,Ks2) #,betaM)

def calc_Ks_wg(Uin,zm = False):
    ## Calculate BetaM
    ## Hoskins and Karoly (see also Vallis (page 551) and Petoukhov et al 2013
    ## and Hoskins and Ambrizzi (1993))

    OMEGA = 7.2921E-5
    a = rearth
    try:
        lats_r = np.deg2rad(Uin.latitude)
    except AttributeError:
        lats_r = np.deg2rad(Uin.lat)

    coslat = np.cos(lats_r)

    betaM1 = 2.0 * OMEGA * coslat * coslat / a

    if zm:
        U = xr.DataArray(Uin,coords={'time':Uin.time,
                                                'latitude':Uin.latitude},
                                          dims = ('time','latitude'))
        
    else:
        U = xr.DataArray(Uin,coords={'time':Uin.time,
                                                'latitude':Uin.latitude,
                                                'longitude':Uin.longitude},
                                          dims = ('time','latitude','longitude'))

    Um = U / coslat

    cos2Um = Um * coslat * coslat

    # first differentiation
    ddy_1 = ddy_merc(cos2Um,zm)
    # divide by cos2phi
    ddy_1_over_cos2p = ddy_1 * (1.0/(coslat * coslat))

    # second differentiation
    ddy_2 = ddy_merc(ddy_1_over_cos2p,zm)

    # this way round so broadcasting gives time as first dimension    
    betaM = - ddy_2 + betaM1
    # Now calculate Ks from BetaM
    Ks2 = a * a * betaM/Um

    Ks = np.sqrt(Ks2)

    return(U,Ks,Ks2) #,betaM)


def ddy_merc(invar,zm=False):
    # based on Hoskins and Karoly:
    # https://journals.ametsoc.org/doi/pdf/10.1175/1520-0469%281981%29038%3C1179%3ATSLROA%3E2.0.CO%3B2
    #  ddy = cos(phi)/a ddphi

    try:
        nlats = len(invar['lat'])
        latname = 'lat'
        lats = invar['lat']
    except KeyError:
        nlats = len(invar['latitude'])
        latname = 'latitude'
        lats = invar['latitude']

    phi = np.deg2rad(lats)
    cosphi = np.cos(phi).values

    dims = invar.shape

    dvardy = invar.copy(deep=True)

    if latname == 'lat':
        dims_var = invar.dims
        latidx_var = dims_var.index('lat')

        dims_lat = invar.lat.dims
        latidx_lat = dims_lat.index('lat')

    elif latname == 'latitude':
        dims_var = invar.dims
        latidx_var = dims_var.index('latitude')

        dims_lat = invar.latitude.dims
        latidx_lat = dims_lat.index('latitude')

    dvar = np.gradient(invar,axis=latidx_var)
    dphi = np.gradient(phi,axis=latidx_lat)

    if zm:
        if len(dims_var) > 2:
            dvardy[...] = (dvar/dphi[:,None]) * (cosphi[:,None] / rearth)
        else:
            dvardy[...] = (dvar/dphi) *(cosphi / rearth)

    else:
        if len(dims_var) > 1:
            dvardy[...] = (dvar/dphi[:,None]) * (cosphi[:,None] / rearth)
        else:
            dvardy[...] = (dvar/dphi) *(cosphi / rearth)

    return(dvardy)


