#%%
import numpy as np
import pandas as np
import xarray as xr
import matplotlib.pyplot as plt
from global_land_mask import globe
import matplotlib as mpl
#%%
lon_min = -100
lon_max = -78
lat_min = 17
lat_max = 31
yr_min = 2003
yr_max = 2021
time_min = np.datetime64('2003-01-01')
time_max = np.datetime64('2020-12-31')
#%% Read SOCAT Coastal
file_socat = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/socat/SOCATv2022_qrtrdeg_gridded_coast_monthly.nc'
ds_socat = xr.open_dataset(file_socat)
idx_lon_socat = np.where((ds_socat.xlon.values>=lon_min) & (ds_socat.xlon.values<=lon_max))[0]
idx_lat_socat = np.where((ds_socat.ylat.values>=lat_min) & (ds_socat.ylat.values<=lat_max))[0]
idx_time_socat = np.where((ds_socat.tmnth.values>=time_min) & (ds_socat.tmnth.values<=time_max))[0]
lon_socat = ds_socat.xlon[idx_lon_socat].values
lat_socat = ds_socat.ylat[idx_lat_socat].values

lat_msh,lon_msh = np.meshgrid(lat_socat,lon_socat,indexing='ij')
x1 = -87.0
y1 = 21.5
x2 = -81
y2 =25.5
a = (y2-y1)/(x2-x1)
b = y2-a*x2
path_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/OISSTv2_highres/'
file_sst = path_sst + 'sst.day.mean.'+ str(yr_min) + '.v2.nc'
ds_i = xr.open_dataset(file_sst)
idx_lon_sst = np.where((ds_i.lon.values>=lon_min+360) & (ds_i.lon.values<=lon_max+360))[0]
idx_lat_sst = np.where((ds_i.lat.values>=lat_min) & (ds_i.lat.values<=lat_max))[0]
lon_sst = ds_i.lon[idx_lon_sst].values-360
lat_sst = ds_i.lat[idx_lat_sst].values
sst_i = np.nanmean(ds_i.sst[:,idx_lat_sst,idx_lon_sst].values,axis=0)

mask_gom = np.ones(lat_msh.shape)
mask_gom[np.isnan(sst_i)] = np.nan
mask_gom[(a*lon_msh+b)>lat_msh] = np.nan

x1 = -83
y1 = 31
x2 = -81
y2 =25.5
a = (y2-y1)/(x2-x1)
b = y2-a*x2
mask_gom[(a*lon_msh+b)<lat_msh] = np.nan

file_topo = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/etopo/ETOPO2v2c_f4.nc'
ds_topo = xr.open_dataset(file_topo)
depth = ds_topo.z.interp(x=lon_socat,y=lat_socat).values
#%%
b_txs_las = -93
b_las_wfs = -87
b_cb_ys = -90

mask_ncoastal = mask_gom.copy()
mask_ncoastal[(lat_msh<24)] = np.nan
mask_ncoastal[(depth<-200)] = np.nan

mask_txs = mask_ncoastal.copy()
mask_txs[lon_msh>=b_txs_las] = np.nan

mask_las = mask_ncoastal.copy()
mask_las[(lon_msh<b_txs_las) | (lon_msh>=b_las_wfs)] = np.nan

mask_wfs = mask_ncoastal.copy()
mask_wfs[(lon_msh<b_las_wfs)] = np.nan

mask_nopen = mask_gom.copy()
mask_nopen[(lat_msh<24)] = np.nan
mask_nopen[(depth>=-200)] = np.nan

mask_nopenw = mask_nopen.copy()
mask_nopenw[lon_msh>=b_txs_las] = np.nan

mask_nopenm = mask_nopen.copy()
mask_nopenm[(lon_msh<b_txs_las) | (lon_msh>=b_las_wfs)] = np.nan

mask_nopenm1 = mask_nopenm.copy()
mask_nopenm1[(lon_msh>=b_cb_ys)] = np.nan
mask_nopenm2 = mask_nopenm.copy()
mask_nopenm2[(lon_msh<b_cb_ys)] = np.nan

mask_nopene = mask_nopen.copy()
mask_nopene[(lon_msh<b_las_wfs)] = np.nan

mask = dict({'ncoastal':mask_ncoastal,
            'txs':mask_txs,
            'las':mask_las,
            'wfs':mask_wfs,
            'nopen':mask_nopen,
            'nopenw':mask_nopenw,
            'nopenm':mask_nopenm,
            'nopene':mask_nopene,
            'nopenm1':mask_nopenm1,
            'nopenm2':mask_nopenm2,})
keys = ['txs','las','wfs','nopenw','nopenm1','nopenm2','nopene']
k = 1.0
for key in keys:
    plt.pcolor(np.ndarray.flatten(lon_socat),np.ndarray.flatten(lat_socat),np.ndarray.flatten(mask[key]))
# 
# %% CCMP 
file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y2003/M01/CCMP_Wind_Analysis_200301_V02.0_L3.5_RSS.nc'
ds_wind = xr.open_dataset(file_wind)
lon_wind = ds_wind.longitude.values
lat_wind = ds_wind.latitude.values
idxlon = np.where((lon_wind>=lon_min+360) & (lon_wind<=lon_max+360))[0]
idxlat = np.where((lat_wind>=lat_min) & (lat_wind<=lat_max))[0]
lon_wind = lon_wind[idxlon]-360.0
lat_wind = lat_wind[idxlat]
uwnd = np.full(((2019-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
vwnd = np.full(((2019-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
wspd = np.full(((2019-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
for idxyr in range(yr_min, 2019):
    for idxmon in range(1,13):
        file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        ds_wind = xr.open_dataset(file_wind)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        uwnd[idx_i,:,:] = ds_wind.uwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        vwnd[idx_i,:,:] = ds_wind.vwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        wspd[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values

idx_sp = np.ndarray.flatten(np.array([np.arange(3,204,12)-1,np.arange(4,204,12)-1,np.arange(5,204,12)-1]).T)
idx_sm = np.ndarray.flatten(np.array([np.arange(6,204,12)-1,np.arange(7,204,12)-1,np.arange(8,204,12)-1]).T)
idx_fl = np.ndarray.flatten(np.array([np.arange(9,204,12)-1,np.arange(10,204,12)-1,np.arange(11,204,12)-1]).T)
idx_wt = np.ndarray.flatten(np.array([np.arange(1,204,12)-1,np.arange(2,204,12)-1,np.arange(12,205,12)-1]).T)

uwnd_sp_avg = np.nanmean(uwnd[idx_sp,:,:],axis=0)
uwnd_sm_avg = np.nanmean(uwnd[idx_sm,:,:],axis=0)
uwnd_fl_avg = np.nanmean(uwnd[idx_fl,:,:],axis=0)
uwnd_wt_avg = np.nanmean(uwnd[idx_wt,:,:],axis=0)

vwnd_sp_avg = np.nanmean(vwnd[idx_sp,:,:],axis=0)
vwnd_sm_avg = np.nanmean(vwnd[idx_sm,:,:],axis=0)
vwnd_fl_avg = np.nanmean(vwnd[idx_fl,:,:],axis=0)
vwnd_wt_avg = np.nanmean(vwnd[idx_wt,:,:],axis=0)

wspd_avg2 = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
wspd_avg2[0,:,:] = np.sqrt(uwnd_sp_avg**2 + vwnd_sp_avg**2)
wspd_avg2[1,:,:] = np.sqrt(uwnd_sm_avg**2 + vwnd_sm_avg**2)
wspd_avg2[2,:,:] = np.sqrt(uwnd_fl_avg**2 + vwnd_fl_avg**2)
wspd_avg2[3,:,:] = np.sqrt(uwnd_wt_avg**2 + vwnd_wt_avg**2)

wspd_avg = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
wspd_avg[0,:,:] = np.nanmean(wspd[idx_sp,:,:],axis=0)
wspd_avg[1,:,:] = np.nanmean(wspd[idx_sm,:,:],axis=0)
wspd_avg[2,:,:] = np.nanmean(wspd[idx_fl,:,:],axis=0)
wspd_avg[3,:,:] = np.nanmean(wspd[idx_wt,:,:],axis=0)
# %%
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig, axs = plt.subplots(2, 4,figsize=(15,7))
for idx in range(1,5):
    plt.subplot(2,4,idx)
    plt.pcolor(lon_socat,lat_socat,wspd_avg2[idx-1,:,:],clim=[0,8],cmap='RdYlBu_r')
    ax = plt.contour(lon_socat,lat_socat,wspd_avg2[idx-1,:,:],clim=[0,8],colors='k')
    plt.clabel(ax,inline=1,fontsize=10)
    plt.text(-101,31.5,chr(96+idx) + ', sqrt(avg(u)$^2$ + avg(v)$^2$)',fontdict={'size':14})
for idx in range(5,9):
    plt.subplot(2,4,idx)
    axp = plt.pcolor(lon_socat,lat_socat,wspd_avg[idx-5,:,:],clim=[0,8],cmap='RdYlBu_r')
    ax = plt.contour(lon_socat,lat_socat,wspd_avg[idx-5,:,:],clim=[0,8],colors='k')
    plt.clabel(ax,inline=1,fontsize=10)
    plt.text(-101,31.5,chr(96+idx) + ', avg(sqrt(u$^2$ + v$^2$))',fontdict={'size':14})
    
# cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat])
fig.colorbar(axp, ax=axs[:,3], shrink=0.6)
# plt.savefig('figs/seasonal_windspeed.png',dpi=300)
#%% 
# %%
