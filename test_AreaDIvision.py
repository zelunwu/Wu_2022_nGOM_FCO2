# %%
import numpy as np
import xarray as xr
import pandas as pd
from glob import glob
from sklearn.cluster import KMeans
from global_land_mask import globe
from mat73 import loadmat
from minisom import MiniSom    
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# %%
lon_min = -100
lon_max = -78
lat_min = 17
lat_max = 31
yr_min = 2003
yr_max = 2021
time_min = np.datetime64('2003-01-01')
time_max = np.datetime64('2020-12-31')
# %% Read SOCAT Coastal
file_socat = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/socat/SOCATv2022_qrtrdeg_gridded_coast_monthly.nc'
ds_socat = xr.open_dataset(file_socat)
idx_lon_socat = np.where((ds_socat.xlon.values>=lon_min) & (ds_socat.xlon.values<=lon_max))[0]
idx_lat_socat = np.where((ds_socat.ylat.values>=lat_min) & (ds_socat.ylat.values<=lat_max))[0]
idx_time_socat = np.where((ds_socat.tmnth.values>=time_min) & (ds_socat.tmnth.values<=time_max))[0]
lon_socat = ds_socat.xlon[idx_lon_socat].values
lat_socat = ds_socat.ylat[idx_lat_socat].values
ds_dist2coast = xr.open_dataset('/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/dist2coast.nc')
dist2coast = ds_dist2coast.d.interp(lon=lon_socat,lat=lat_socat).values
mask_coast = np.float64(dist2coast<=375)
mask_coast[mask_coast<0.1] = np.nan
lat_socat_msh,lon_socat_msh = np.meshgrid(lat_socat,lon_socat,indexing='ij')
mask_ocean = np.float64(globe.is_ocean(lat_socat_msh,lon_socat_msh))
mask_ocean[mask_ocean<0.5] = np.nan
mask_coast = mask_coast*mask_ocean

lat_msh,lon_msh = np.meshgrid(lat_socat,lon_socat,indexing='ij')
x1 = -87.0
y1 = 21.5
x2 = -81
y2 =25.5
a = (y2-y1)/(x2-x1)
b = y2-a*x2
mask_gom = mask_ocean.copy()
mask_gom[(a*lon_msh+b)>lat_msh] = np.nan

x1 = -83
y1 = 31
x2 = -81
y2 =25.5
a = (y2-y1)/(x2-x1)
b = y2-a*x2
mask_gom[(a*lon_msh+b)<lat_msh] = np.nan
# %% SST data
path_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/OISSTv2_highres/'
file_sst = path_sst + 'sst.day.mean.'+ str(yr_min) + '.v2.nc'
ds_i = xr.open_dataset(file_sst)
idx_lon_sst = np.where((ds_i.lon.values>=lon_min+360) & (ds_i.lon.values<=lon_max+360))[0]
idx_lat_sst = np.where((ds_i.lat.values>=lat_min) & (ds_i.lat.values<=lat_max))[0]
lon_sst = ds_i.lon[idx_lon_sst].values-360
lat_sst = ds_i.lat[idx_lat_sst].values
sst_m = np.full([12*(yr_max-yr_min),len(idx_lat_sst),len(idx_lon_sst)],np.nan)
for idxy in range(yr_min,yr_max):
    file_sst = path_sst + 'sst.day.mean.'+ str(idxy) + '.v2.nc'
    ds_i = xr.open_dataset(file_sst)
    sst_i = ds_i.sst[:,idx_lat_sst,idx_lon_sst].values
    month_i = pd.to_datetime(ds_i.time.values).month
    sst_m_i = np.empty([12,len(idx_lat_sst),len(idx_lon_sst)])
    for idxm in range(1,13):
        idxt = np.where(month_i==idxm)[0]
        sst_m[(idxy-yr_min)*12+idxm-1,:,:] = ds_i.sst[idxt,idx_lat_sst,idx_lon_sst].mean(dim='time').values
sst_m_clim = np.nanmean(np.reshape(sst_m,[yr_max-yr_min,12,len(lat_sst),len(lon_sst)]),axis=0)
sst_m_clim = np.repeat(np.reshape(sst_m_clim,[1,12,len(lat_sst),len(lon_sst)]),yr_max-yr_min,axis=0)
sst_m_clim = np.reshape(sst_m_clim,sst_m.shape)
#%%
mask_gom[np.isnan(np.nanmean(sst_m_clim,axis=0))] = np.nan
# plt.pcolor(lon_sst,lat_sst,mask_gom)
# %% CHLA
path_chla = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/modis/mapped_monthly/chlor_a/'
file_chla = path_chla + 'A20030012003031.L3m_MO_CHL_chlor_a_4km.nc'
ds_i = xr.open_dataset(file_chla)
idx_lon_chla = np.where((ds_i.lon.values>=lon_min) & (ds_i.lon.values<=lon_max))[0]
idx_lat_chla = np.where((ds_i.lat.values>=lat_min) & (ds_i.lat.values<=lat_max))[0]
lon_chla = ds_i.lon[idx_lon_chla].values
lat_chla = ds_i.lat[idx_lat_chla].values
chla_m = np.full([12*(yr_max-yr_min),len(idx_lat_sst),len(idx_lon_sst)],np.nan)
for idxy in range(yr_min,yr_max):
    files_i = np.sort(glob(path_chla+'A'+str(idxy)+'*'))
    for idxm in range(1,13):
        file_i = files_i[idxm-1]
        ds_i = xr.open_dataset(file_i)
        chla_m[(idxy-yr_min)*12+idxm-1,:,:] = ds_i.chlor_a.interp(lon=lon_sst,lat=lat_sst).values
chla_m_clim = np.nanmean(np.reshape(chla_m,[yr_max-yr_min,12,len(lat_sst),len(lon_sst)]),axis=0)
chla_m_clim = np.repeat(np.reshape(chla_m_clim,[1,12,len(lat_sst),len(lon_sst)]),yr_max-yr_min,axis=0)
chla_m_clim = np.reshape(chla_m_clim,chla_m.shape)
# %% ERA5 Wind
file_era5 = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/ERA5/era5_monthly_u10_global.nc'
ds_era5 = xr.open_dataset(file_era5)
idx_lon_era5 = np.where((ds_era5.longitude.values>=lon_min+360) & (ds_era5.longitude.values<=lon_max+360))[0]
idx_lat_era5 = np.where((ds_era5.latitude.values>=lat_min) & (ds_era5.latitude.values<=lat_max))[0]
idx_time_era5 = np.where((ds_era5.time.values>=time_min) & (ds_era5.time.values<=time_max))[0]
lon_era5 = ds_era5.longitude[idx_lon_era5].values-360
lat_era5 = ds_era5.latitude[idx_lat_era5].values
time_era5 = ds_era5.time.values
u10_era5 = ds_era5.si10.interp(longitude=lon_sst+360,latitude=lat_sst).values
u10_era5 = u10_era5[idx_time_era5,0,:,:]
u10_era5_clim = np.nanmean(np.reshape(u10_era5,[int(u10_era5.shape[0]/12),12,u10_era5.shape[1],u10_era5.shape[2]]),axis=0)
# %% Chen 19
# file_chen = './data/pco2_chen.mat'
# data = loadmat(file_chen)
# time_chen = data['time']
# lon_chen = data['lon_1km']
# lat_chen = data['lat_1km']
# pco2_chen = data['pco2_1km']
# idxtimechen = np.where((time_era5>=np.datetime64('2002-07-01')) & (time_era5<=np.datetime64('2017-12-31')))[0]
# time_chen = time_era5[idxtimechen]
# ds_chen = xr.Dataset(
#     data_vars=dict(
#         pco2=(["lon", "lat", "time"], pco2_chen),
#     ),
#     coords=dict(
#         lon=lon_chen,
#         lat=lat_chen,
#         time=time_chen,
#     ),
# )
# ds_chen.to_netcdf('./data/pco2_chen.nc')
ds_chen = xr.open_dataset('./data/pco2_chen.nc')
pco2_chen = np.transpose(ds_chen.pco2.interp(lon=lon_sst,lat=lat_sst).values[:,:,6:],[2,1,0])
pco2_chen_m_clim = np.nanmean(np.reshape(pco2_chen,[15,12,len(lat_sst),len(lon_sst)]),axis=0)
pco2_chen_m_clim = np.repeat(np.reshape(pco2_chen_m_clim,[1,12,len(lat_sst),len(lon_sst)]),15,axis=0)
pco2_chen_m_clim = np.reshape(pco2_chen_m_clim,pco2_chen.shape)
# %%
lat_msh,lon_msh = np.meshgrid(lat_sst,lon_sst,indexing='ij')
# Clim 
sst_clim = np.nanmean(sst_m_clim,axis=0)
chla_clim = np.nanmean(np.log10(chla_m_clim),axis=0)
u10_clim = np.nanmean(u10_era5_clim,axis=0)
pco2_clim = np.nanmean(pco2_chen_m_clim,axis=0)
# STD
sst_std = np.nanstd(sst_m_clim,axis=0)
chla_std = np.nanstd(np.log10(chla_m_clim),axis=0)
u10_std = np.nanstd(u10_era5_clim,axis=0)
pco2_std = np.nanstd(pco2_chen_m_clim,axis=0)

X = np.array([np.ndarray.flatten(lon_msh*mask_coast),
              np.ndarray.flatten(lat_msh*mask_coast),
              np.ndarray.flatten(sst_clim*mask_coast),
              np.ndarray.flatten(chla_clim*mask_coast),
              np.ndarray.flatten(u10_clim*mask_coast),
              np.ndarray.flatten(sst_std*mask_coast),
              np.ndarray.flatten(chla_std*mask_coast),
              np.ndarray.flatten(u10_std*mask_coast),])
df_X = pd.DataFrame({'lon':np.ndarray.flatten(lon_msh*mask_gom),
                     'lat':np.ndarray.flatten(lat_msh*mask_gom),
                     'sst_clim':np.ndarray.flatten(sst_clim*mask_gom),
                     'sst_std':np.ndarray.flatten(sst_std*mask_gom),
                     'chla_clim':np.ndarray.flatten(chla_clim*mask_gom),
                     'chla_std':np.ndarray.flatten(chla_std*mask_gom),
                     'u10_clim':np.ndarray.flatten(u10_clim*mask_gom),
                     'u10_std':np.ndarray.flatten(u10_std*mask_gom),
                     'pco2_clim':np.ndarray.flatten(pco2_clim*mask_gom),
                     'pco2_std':np.ndarray.flatten(pco2_std*mask_gom),})
df_X = df_X.dropna()
df_X.index = range(df_X.shape[0])
df_mean = df_X.mean()
df_anom = df_X-df_mean
df_std = df_X.std()
df_norm = df_anom/df_std
# %%
data = df_norm.values[:,2:]
inertia = []
sil_score = []
for k in range(2,50):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(data)
    inertia.append(np.sqrt(kmeans.inertia_))
    sil_score.append(silhouette_score(data,kmeans.labels_))
plt.plot(range(2,50),inertia,'o-')
plt.xlabel('k')
# idx_loc = np.where((df_X.lon.values>-100) &(df_X.lat.values<50))[0]
# ydata = kmeans.fit_predict(df_norm.values[:,-2:])


# %%
# plt.scatter(df_X.lon,df_X.lat,20,labels)
kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(data)
labels = kmeans.labels_
#%% 
label_new = labels.copy()
label_new[(labels==3) & (df_X.lon.values<-90)] = 10
label_new[(labels==1) & (df_X.lat.values<22)] = 10
label_new[(labels==0) & (df_X.lat.values<23)& (df_X.lon.values>-92)] = 2
label_new[(labels==0) & (df_X.lon.values>-86)] = 3
label_new[(labels==4) & (df_X.lat.values>=28)] = 6
label_new[(labels==4) & (df_X.lat.values<=28)] = 1
label_new[(labels==5) & (df_X.lon.values<=-87)] = 6
label_new[(labels==5) & (df_X.lon.values>-87)] = 11
label_new[(labels==6) & (df_X.lon.values>-87)] = 11
# label_new[(labels==6) & (df_X.lon.values<=-96)] = 8
#%%
for label in np.arange(0,10):
    idxlabel = np.where(labels == label)[0]
    plt.scatter(df_X.lon[idxlabel],df_X.lat[idxlabel],20)
plt.colorbar()
# %%
data = df_norm.values[:,-2:]
som_shape = (4, 4)
som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=0.3, learning_rate=0.5) # initialization of 6x6 SOM
som.train(data, 100, verbose=True) # trains the SOM with 100 iterations

winner_coordinates = np.array([som.winner(x) for x in data]).T
cluster_index = np.ravel_multi_index(winner_coordinates, som_shape)
print('Total provinces = ' + str(len(np.unique(cluster_index))))
for c in np.unique(cluster_index):
    idx = np.where(cluster_index == c)[0]
    plt.scatter(df_X.lon[idx].values,df_X.lat[idx].values,20)
# %%
