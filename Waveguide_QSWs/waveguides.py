# Module to identify waveguides in refractive index data
# Written by rhwhite rachel.white@cantab.net
import numpy as np
import xarray as xr
import math
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from datetime import date

# Look for waveguides by searching for turning points at specific wavenumbers
def iden_waveguide_TPs_map(inKs2,inU,Uthresh,minwidth,mindepth,wnstart,wnend,nwgs,toprint=False):
    # Check that latitudes are the correct way round!
    if inKs2.latitude[0] > inKs2.latitude[1]:
        exit('latitudes must be south to north')

    TPs = {}
    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1])
    WGdepth = np.ndarray([nwgs,wnend - wnstart+1])
    WGwidth = np.ndarray([nwgs,wnend - wnstart+1])
   
    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan
    WGdepth[...] = np.nan
    WGwidth[...] = np.nan

    # define waveguide map so latitude indexing works easily
    nlats = len(inKs2.latitude)
    nk = wnend - wnstart+1

    wg_map = xr.DataArray(np.ndarray([nk,nlats]),coords={
                       'k':np.arange(wnstart,wnend+1),
                       'latitude':inKs2.latitude},
                        dims=('k','latitude'))

    wg_map[...] = 0

    if np.any(np.isfinite(inKs2.values)): # only if there are some non-nan values

        # For each wavenumber find the turning points
        for iwn in range(wnstart,wnend+1):
            if toprint: print('wavenumber:' + str(iwn))
            Ks2_wn = (inKs2 - iwn**2)
            # Find local minima
            TPs[iwn] = np.where(np.diff(np.sign(Ks2_wn)))[0]
            # these are always the points BEFORE the change in sign
            # So for TP2, we want to add 1 to the index
            if toprint: print(TPs[iwn])

            # For each pair of TPs, check if it's a waveguide
            iwg = 0
            nTPs = len(TPs[iwn])
            for iTP in np.arange(0,nTPs-1):
                if toprint:
                    print('turning point ' + str(iTP))
                # check Ks2 exceeds threshold everywhere between TPs:
                # Add 1 from first TP because by definition the values AT the 
                # turning points are less than or equal to the iwn**2
                # Add 1 from second TP because slice doesn't include the last value 
                if ~np.any(inKs2[TPs[iwn][iTP]+1:TPs[iwn][iTP+1]+1].values < iwn**2):
		    #Set inside edge TPs for this waveguide:
                    TP1 = TPs[iwn][iTP] + 1 # add 1 as the code find the point BEFORE it turns INTO waveguide
                    TP2 = TPs[iwn][iTP + 1] # don't add one, as this is BEFORE it turns OUT of waveguide
                    
                    # Calculate waveguide depth in Ks2
                    max_Ks2 = np.amax(inKs2[TP1:TP2 + 1].values) # add 1 to include last value
                    if toprint:
                        print("Ks2 and threshold:")
                        print(max_Ks2)
                        print(iwn+mindepth)
                    # Apply waveguide depth critera:
                    if max_Ks2 >= (iwn+mindepth)**2:
                        # U threshold criteria - WITHIN waveguide (i.e. don't include outside edges)
                        if toprint: 
                            print('U values for threshold check')
                            print(inU[TP1:TP2+1].values)
                        if ~np.any(inU[TP1:TP2+1].values < Uthresh):

                            # Width of waveguide criteria, in degrees latitude:
                            # Strict definition: width of waveguide is at least this width
                            TPlat1 = inKs2.latitude[TP1].values
                            TPlat2 = inKs2.latitude[TP2].values # 

                            if np.abs(TPlat1 - TPlat2) >= minwidth:
                                if toprint: 
                                    print('WAVEGUIDE!!')
                                    print('TPs:')
                                    print(TPlat1,TPlat2)
                                    print((TPlat1 - TPlat2))
                                WGlatmin[iwg,iwn-wnstart] = TPlat1
                                WGlatmax[iwg,iwn-wnstart] = TPlat2
                                WGdepth[iwg,iwn-wnstart] = np.sqrt(max_Ks2) - iwn
                                WGwidth[iwg,iwn-wnstart] = np.abs(TPlat1 - TPlat2)

                                iwg +=1 

                                # .sel includes the end value, so we don't need to add anything here (isel does not)
                                Ks2_inwg = inKs2.sel(latitude = slice(TPlat1,TPlat2))
                                depth = np.sqrt(Ks2_inwg) - iwn
                                if np.any(depth < 0):
                                    exit('something has gone wrong, depth is negative')
                                if toprint: print(TPlat1,TPlat2)
                                
                                wg_map.sel(k=iwn).sel(latitude=slice(TPlat1,TPlat2)).values[...] = depth
                                    
                                
                            else:
                                if toprint: print('not a waveguide, didn\'t pass width test')
                        else:
                            if toprint: print('not a waveguide, didn\'t pass U threshold test')

                else:
                    if toprint: print('not a waveguide, didn\'t pass Ks exceeds everywhere threshold')

    return(WGlatmin,WGlatmax,WGdepth,WGwidth,wg_map.values)

def calc_waveguide_map(Uin,Ksin,Ks2in,lat1,lat2,
                        Uthresh,wnstart,wnend,minwidth,mindepth,
                        Uname,interp =False,toprint=False):


    # Check which way the latitudes are and flip indata if necessary
    if Ks2in.isel(latitude=0).latitude.values > Ks2in.isel(latitude=1).latitude.values:
        Ks2in = Ks2in.isel(latitude=slice(None, None, -1))
    if Ksin.isel(latitude=0).latitude.values > Ksin.isel(latitude=1).latitude.values:
        Ksin = Ksin.isel(latitude=slice(None, None, -1))
    if Uin.isel(latitude=0).latitude.values > Uin.isel(latitude=1).latitude.values:
        Uin = Uin.isel(latitude=slice(None, None, -1))

    # Set size of waveguide arrays 
    ntimes = len(Ks2in.time) # number of timesteps
    try:
        nlons = len(Ks2in.longitude) # number of longitudes
        ZM=False
    except AttributeError:
        ZM=True
        nlons=1

    nwgs = 10 # maximum number of waveguides that can exist at any longitude:
    nk = wnend - wnstart+1 # number of wavenumbers to look for waveguides for
    nlats = len(Ks2in.latitude.sel(latitude = slice(lat1,lat2))) # latitudes

    # define arrays
    WGlatmin = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGlatmax = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGdepth = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])
    WGwidth = np.ndarray([nwgs,wnend - wnstart+1,nlons,ntimes])

    WGlatmin[...] = np.nan
    WGlatmax[...] = np.nan
    WGdepth[...] = np.nan
    WGwidth[...] = np.nan

    WGmap = np.ndarray([ntimes,nk,nlats,nlons])

    # define map as xarray so we can reference and fill latitudes more easily
    xr_wg_map = xr.DataArray(WGmap,coords={
                       'time':Ks2in.time,'k':np.arange(wnstart,wnend+1),
                       'longitude':Ks2in.longitude,'latitude':Ks2in.latitude.sel(latitude = slice(lat1,lat2))},
                        dims=('time','k','latitude','longitude'))
    xr_wg_map[...] = 0
    
    for timein in np.arange(0,ntimes):
        if timein%100 == 0: print(timein)

        for ilon in np.arange(0,nlons):
            if toprint: print('longitude: ' + str(ilon))
            if ZM: # then zonal mean
                Ks2_sel = Ks2in.isel(time=timein).sel(latitude=slice(lat1,lat2)).copy(deep=True)
                U_sel = Uin.isel(time=timein).sel(latitude=slice(lat1,lat2)).copy(deep=True)

            else: # select longitude
                Ks2_sel = Ks2in.isel(time=timein).isel(longitude=ilon).sel(latitude = slice(lat1,lat2)).squeeze()
                U_sel = Uin.isel(time=timein).isel(longitude=ilon).sel(latitude = slice(lat1,lat2)).squeeze()

                
            (WGlatmin[:,:,ilon,timein],WGlatmax[:,:,ilon,timein],WGdepth[:,:,ilon,timein],WGwidth[:,:,ilon,timein],
                xr_wg_map.isel(time=timein).isel(longitude=ilon).values[...]) = iden_waveguide_TPs_map(Ks2_sel,U_sel,
                    Uthresh=Uthresh,minwidth=minwidth,mindepth=mindepth,
                    wnstart=wnstart,wnend=wnend,nwgs=nwgs,toprint=toprint)
            
    ## Make dataset
    if ZM:
        longitudes_out = [0]
    else:
        longitudes_out = Ks2in.longitude

    xr_wg_minlat = xr.DataArray(WGlatmin,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))

    xr_wg_maxlat = xr.DataArray(WGlatmax,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    

    xr_wg_depth = xr.DataArray(WGdepth,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    
    xr_wg_width = xr.DataArray(WGwidth,coords={
                       'wg_num':np.arange(1,nwgs+1),'k':np.arange(wnstart,wnend+1),
                       'longitude':longitudes_out,'time':Ks2in.time},
                       dims=('wg_num','k','longitude','time'))    
             
    # Convert to datasets and append
    ds = xr_wg_minlat.to_dataset(name = 'WG_min_lat')
    ds['WG_max_lat'] = xr_wg_maxlat
    ds['WG_depth'] = xr_wg_depth
    ds['WG_width'] = xr_wg_width

    ds.attrs['history'] = ('Created on ' + str(date.today()) + ' by calc_waveguide_map in rhwhitepackages3')
    ds.attrs['input data'] = ('Input files = ' + Uname)
    ds.attrs['parameters'] = ('waveguide minimum width = ' + str(minwidth) + '; minimum zonal wind in waveguide = ' + str(Uthresh) +
                              '; waveguide minimum depth = ' + str(mindepth))

    dsmap = xr_wg_map.to_dataset(name = 'WG_map')
    dsmap.attrs['history'] = ('Created on ' + str(date.today()) + ' by calc_waveguide_map in rhwhitepackages3')
    dsmap.attrs['input data'] = ('Input files = ' + Uname)
    dsmap.attrs['parameters'] = ('waveguide minimum width = ' + str(minwidth) + '; minimum zonal wind in waveguide = ' + str(Uthresh) +
                              '; waveguide minimum depth = ' + str(mindepth))

    return(ds,dsmap)



def calc_waveguide_clims(hemi,plev,dirin,WGmap_filein):
    # Read in waveguide maps
    WGmap = xr.open_dataset(dirin + '/' + WGmap_filein + '.nc')

    # Remove leapyears
    datain = WGmap.sel(time=slice('1980','2022'))
    datain_noleap = datain.sel(time=~((datain.time.dt.month == 2) & (datain.time.dt.day == 29)))

    # Check that we have full years:
    if (len(datain_noleap.time)%365) != 0:
        exit('need to do something else to make sure this is full years')
 
    # convert to climatology through re-shaping and taking mean
    nyears = int(len(datain_noleap.time)/365)
        
    daily_WGmap_NL_clim = xr.DataArray(np.reshape(datain_noleap.WG_map.values,[nyears,365,len(datain_noleap.k),
                                                    len(datain_noleap.latitude),len(datain_noleap.longitude)]).mean(axis=0),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})


    # Tile to wrap around DJF
    daily_WGmap_NL_clim_rep = xr.DataArray(np.tile(daily_WGmap_NL_clim.values,[3,1,1,1]),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365*3),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})


    # created smoothed climatology at each point
    daily_WGmap_NL_clim_smooth = daily_WGmap_NL_clim.copy(deep=True)

    for ilat in range(0,len(daily_WGmap_NL_clim.latitude)):
        for ilon in range(0,len(daily_WGmap_NL_clim.longitude)):
            for ik in range(0,len(daily_WGmap_NL_clim.k)):
                filtered = savgol_filter(daily_WGmap_NL_clim_rep[:,ik,ilat,ilon],window_length=61, polyorder=1)
                daily_WGmap_NL_clim_smooth[:,ik,ilat,ilon] = filtered[365:365+365]


    daily_WGmap_NL_clim_smooth= daily_WGmap_NL_clim_smooth.to_dataset(name='WGmap_clim_smooth')
    daily_WGmap_NL_clim_smooth.attrs["description"] = ("WGmap 1980-2022 smooth climatology with savgol_filter" + 
                                "with window length = 61 and polyorder = 1, wrapped to create continuity in DJF")

    daily_WGmap_NL_clim_smooth.to_netcdf(dirin + '/' + WGmap_filein + '_smooth_clim.nc')
    # tile and subtract to get anomalies

    daily_WGmap_NL_clim_anoms = datain_noleap.WG_map - np.tile(daily_WGmap_NL_clim_smooth.WGmap_clim_smooth,(nyears,1,1,1))
    daily_WGmap_NL_clim_anoms.to_netcdf(dirin + '/' + WGmap_filein + 'anoms_from_smooth_clim.nc')

    # Repeat with binary yes/no to get climatological frequency
    datain_noleap_freq = np.where(datain_noleap.WG_map>0,1,0)

    daily_WGmap_NL_clim_freq = xr.DataArray(np.reshape(datain_noleap_freq,[nyears,365,len(datain_noleap.k),
                                                    len(datain_noleap.latitude),len(datain_noleap.longitude)]).mean(axis=0),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})

    daily_WGmap_NL_clim_freq= daily_WGmap_NL_clim_freq.to_dataset(name='WGmap_freq_clim')
    daily_WGmap_NL_clim_freq.attrs["description"] = ("WGmap frequency 1980-2022 climatology")

    daily_WGmap_NL_clim_freq.to_netcdf(dirin + '/' + WGmap_filein + '_freq_clim.nc')


    # Tile to wrap around DJF
    daily_WGmap_NL_clim_freq_rep = xr.DataArray(np.tile(daily_WGmap_NL_clim_freq.values,[3,1,1,1]),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365*3),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})

    # created smoothed climatology at each point
    daily_WGmap_NL_clim_freq_smooth = daily_WGmap_NL_clim_freq.copy(deep=True)

    for ilat in range(0,len(daily_WGmap_NL_clim_freq.latitude)):
        for ilon in range(0,len(daily_WGmap_NL_clim_freq.longitude)):
            for ik in range(0,len(daily_WGmap_NL_clim_freq.k)):
                filtered = savgol_filter(daily_WGmap_NL_clim_freq_rep[:,ik,ilat,ilon],
                                    window_length=61, polyorder=1)
                daily_WGmap_NL_clim_freq_smooth[:,ik,ilat,ilon] = filtered[365:365+365]


    daily_WGmap_NL_clim_freq_smooth= daily_WGmap_NL_clim_freq_smooth.to_dataset(name='WGmap_freq_clim_smooth')
    daily_WGmap_NL_clim_freq_smooth.attrs["description"] = ("WGmap frequency 1980-2022 smooth climatology with savgol_filter" + 
                                "with window length = 61 and polyorder = 1, wrapped to create continuity in DJF")

    daily_WGmap_NL_clim_freq_smooth.to_netcdf(dirin + '/' + WGmap_filein + '_freq_smooth_clim.nc')


def calc_waveguide_clims_year(hemi,plev,dirin,WGmap_filein,startyr,endyr):
    # Read in waveguide maps
    WGmap = xr.open_dataset(dirin + '/' + WGmap_filein + '.nc')

    # Remove leapyears
    datain = WGmap.sel(time=slice(startyr,endyr))
    datain_noleap = datain.sel(time=~((datain.time.dt.month == 2) & (datain.time.dt.day == 29)))

    # Check that we have full years:
    if (len(datain_noleap.time)%365) != 0:
        exit('need to do something else to make sure this is full years')

    # convert to climatology through re-shaping and taking mean
    nyears = int(len(datain_noleap.time)/365)

    daily_WGmap_NL_clim = xr.DataArray(np.reshape(datain_noleap.WG_map.values,[nyears,365,len(datain_noleap.k),
                                                    len(datain_noleap.latitude),len(datain_noleap.longitude)]).mean(axis=0),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})

   # Tile to wrap around DJF
    daily_WGmap_NL_clim_rep = xr.DataArray(np.tile(daily_WGmap_NL_clim.values,[3,1,1,1]),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365*3),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})

    # created smoothed climatology at each point
    daily_WGmap_NL_clim_smooth = daily_WGmap_NL_clim.copy(deep=True)

    for ilat in range(0,len(daily_WGmap_NL_clim.latitude)):
        for ilon in range(0,len(daily_WGmap_NL_clim.longitude)):
            for ik in range(0,len(daily_WGmap_NL_clim.k)):
                filtered = savgol_filter(daily_WGmap_NL_clim_rep[:,ik,ilat,ilon],window_length=61, polyorder=1)
                daily_WGmap_NL_clim_smooth[:,ik,ilat,ilon] = filtered[365:365+365]


    daily_WGmap_NL_clim_smooth= daily_WGmap_NL_clim_smooth.to_dataset(name='WGmap_clim_smooth')
    daily_WGmap_NL_clim_smooth.attrs["description"] = ("WGmap 1980-2022 smooth climatology with savgol_filter" +
                                "with window length = 61 and polyorder = 1, wrapped to create continuity in DJF")

    daily_WGmap_NL_clim_smooth.to_netcdf(dirin + '/' + WGmap_filein + '_smooth_clim.nc')
    # tile and subtract to get anomalies

    daily_WGmap_NL_clim_anoms = datain_noleap.WG_map - np.tile(daily_WGmap_NL_clim_smooth.WGmap_clim_smooth,(nyears,1,1,1))
    daily_WGmap_NL_clim_anoms.to_netcdf(dirin + '/' + WGmap_filein + 'anoms_from_smooth_clim.nc')

    # Repeat with binary yes/no to get climatological frequency
    datain_noleap_freq = np.where(datain_noleap.WG_map>0,1,0)

    daily_WGmap_NL_clim_freq = xr.DataArray(np.reshape(datain_noleap_freq,[nyears,365,len(datain_noleap.k),
                                                    len(datain_noleap.latitude),len(datain_noleap.longitude)]).mean(axis=0),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})
    # Tile to wrap around DJF
    daily_WGmap_NL_clim_freq_rep = xr.DataArray(np.tile(daily_WGmap_NL_clim_freq.values,[3,1,1,1]),
                                                     dims=('time','k','latitude','longitude'),
                                                     coords={'time':np.arange(0,365*3),
                                                             'k':datain_noleap.k,
                                                             'latitude':datain_noleap.latitude,
                                                             'longitude':datain_noleap.longitude})

    # created smoothed climatology at each point
    daily_WGmap_NL_clim_freq_smooth = daily_WGmap_NL_clim_freq.copy(deep=True)

    for ilat in range(0,len(daily_WGmap_NL_clim_freq.latitude)):
        for ilon in range(0,len(daily_WGmap_NL_clim_freq.longitude)):
            for ik in range(0,len(daily_WGmap_NL_clim_freq.k)):
                filtered = savgol_filter(daily_WGmap_NL_clim_freq_rep[:,ik,ilat,ilon],
                                    window_length=61, polyorder=1)
                daily_WGmap_NL_clim_freq_smooth[:,ik,ilat,ilon] = filtered[365:365+365]


    daily_WGmap_NL_clim_freq_smooth= daily_WGmap_NL_clim_freq_smooth.to_dataset(name='WGmap_freq_clim_smooth')
    daily_WGmap_NL_clim_freq_smooth.attrs["description"] = ("WGmap frequency 1980-2022 smooth climatology with savgol_filter" +
                                "with window length = 61 and polyorder = 1, wrapped to create continuity in DJF")

    daily_WGmap_NL_clim_freq_smooth.to_netcdf(dirin + '/' + WGmap_filein + '_freq_smooth_clim.nc')

