# %%
from cProfile import label
from json import load
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from global_land_mask import globe
import matplotlib as mpl
import pandas as pd
from glob import glob
from PyCO2SYS import sys
from scipy.io import loadmat
import warnings
warnings.filterwarnings("ignore")
import datetime as dt

from datetime import datetime
from copy import deepcopy
import math
import re
import warnings
import os

# 3rd party libraries
import statsmodels.api as sm
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
import numpy as np
# %%
def separator_check(text):
    for sep in [",", "\t"]:
        if re.search(sep, text):
            return sep
    return None

# functions to load data file to dataframe
def load_file_to_pandas(file_name, comment_indicator=None):
    with open(file_name, "r") as f:
        sep = None
        for data in f:
            sep = separator_check(data)
            break
    return pd.read_csv(file_name, sep=sep, engine="python", dtype={0: str})

def trend_options(df_columns):
    print("")
    for i, col in enumerate(df_columns[1:]):
        print("{}:\t{}".format(i + 1, col))
    selection = int(
        input("Select the number associated with the parameter to process for trends: ")
    )
    print("{} selected...".format(df_columns[selection]))
    print("")
    return df_columns[selection]

class ExtensionException(Exception):
    pass


# Date format test
def build_formatted_string(string: str, char_requirements_dict: dict):
    from collections import Counter

    string = np.array(list(string.lower()))
    _, idx = np.unique(string, return_index=True)
    char_counter = Counter(string)
    string = string[np.sort(idx)]
    output = []

    # check for required characters
    if "y" in char_requirements_dict:
        for c in char_requirements_dict.keys():
            if c not in char_counter:
                raise FormatError(
                    f"The character {c} is missing in your date format and is required."
                )
    elif "h" in char_requirements_dict:
        if "h" not in char_counter:
            raise FormatError(
                "The character h is missing in your time format and is required"
            )

    for c in string:
        # check that there are no meaningless additional characters
        if c not in char_requirements_dict:
            raise FormatError(
                f"{c} is not a valid character for a datetime format or your datetime is ordered incorrectly."
            )
        # make sure number of characters is consistent with formatting constraints
        if char_counter[c] not in char_requirements_dict[c]:
            raise FormatError(
                f"{c} must have {char_requirements_dict[c]} number of characters."
            )
        elif char_counter[c] == char_requirements_dict[c][0]:
            output.append(c)
        elif char_counter[c] == char_requirements_dict[c][1]:
            output.append(c.upper())
    return "%" + "%".join(output)

def convert_datetime_format_string(string: str):
    date_requirements_dict = {"m": [2], "d": [2], "y": [2, 4]}
    time_requirements_dict = {"h": [None, 2], "m": [None, 2], "s": [None, 2]}
    separated_string = string.split(" ")
    if len(separated_string) > 2:
        raise FormatError(
            "There are too many inputs for a datetime string format. Make sure your datetime only has one space between the date and time."
        )
    output = []
    for i in range(len(separated_string)):
        # assume first part is date, second part is time
        if i == 0:
            output.append(
                build_formatted_string(separated_string[i], date_requirements_dict)
            )
        elif i == 1:
            output.append(
                build_formatted_string(separated_string[i], time_requirements_dict)
            )
        else:
            raise FormatError(
                "There are too many spaces in your date/time format. There should be at most two, one for the date and one for the time."
            )
    return " ".join(output)

class FormatError(Exception):
    pass

def decimal_month(years, months, days):
    """Compute the decimal month (ex. March 15 would be 3.48 or 3 + 15/31). Account for number of days in February using year."""
    days_in_month = {
        1: 31,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }
    feb_days = {
        y: 29 if y % 400 == 0 or (y % 4 == 0 and not y % 100 == 0) else 28
        for y in np.unique(years)
    }
    output = []
    for y, m, d in zip(years, months, days):
        if m == 2:
            output.append(m + (d - 1) / feb_days[y])
        else:
            output.append(m + (d - 1) / days_in_month[m])
    return output


def datapreprocessing(ts_df,dropnan=False):
    ts_df_new = ts_df.copy()
    datetime_col = ts_df_new.columns[0]
    variable_names = list(ts_df_new.columns[1:])
    dayfirst = "n"
    dayformat = None

    try:  # if date is entirely numeric with no separators
        date_test = np.array([t.split() for t in ts_df_new.iloc[:, 0]]).astype(int)
        dayformat = input(
            "What is the date format of your data (ex yyyymmdd, ddmmyyyy HHMMSS, etc)? "
        ).upper()
        dayformat = convert_datetime_format_string(dayformat)

    except ValueError:
        day_test_df = ts_df_new.iloc[:, 0].str.split("[,\s/-]+|T", expand=True)
        # if datetime can be separated. Otherwise, try to let pandas handle it.
        if len(day_test_df.columns) > 2:
            int_columns = []
            # test whether each column in split date can be converted to int
            for col in day_test_df.columns[:3]:
                try:
                    day_test_df = day_test_df.astype({col: int})
                    int_columns.append(col)
                except (
                    ValueError
                ):  # assume an error results from the month in string form
                    pass
            day_test_max = day_test_df.iloc[:, int_columns].aggregate(max)
            daypos = [
                col
                for col in day_test_max.index
                if day_test_max[col] > 12 and day_test_max[col] < 32
            ]
            if daypos and int_columns[daypos[0]] == 0:
                dayfirst = "y"
            elif not dayfirst:
                dayfirst = input(
                    "Are days first in the datetimes being imported (y or n)? "
                )
    except FormatError as e:
        print("The following exception was raised: ", e)
        print(
            "Please rerun the cell with a properly formatted date format or reformat your datetimes in your file to an acceptable format."
        )

    # make datetime the index and calculate decimal year
    if dayformat:
        ts_df_new[datetime_col] = pd.to_datetime(
            ts_df_new[datetime_col], format=dayformat
        )
    elif dayfirst == "y":
        ts_df_new[datetime_col] = pd.to_datetime(ts_df_new[datetime_col], dayfirst=True)
    else:
        ts_df_new[datetime_col] = pd.to_datetime(ts_df_new[datetime_col])

    ts_df_new.index = ts_df_new[datetime_col]
    ts_df_new["year"] = pd.DatetimeIndex(ts_df_new[datetime_col]).year
    ts_df_new["month"] = pd.DatetimeIndex(ts_df_new[datetime_col]).month
    ts_df_new["day"] = pd.DatetimeIndex(ts_df_new[datetime_col]).day
    ts_df_new["decimal_month"] = decimal_month(
        ts_df_new["year"], ts_df_new["month"], ts_df_new["day"]
    )
    ts_df_new["decimal_year"] = (
        ts_df_new["year"] + (ts_df_new["decimal_month"] - 1) / 12
    )

    # remove NaNs
    if dropnan:
        ts_df_new.dropna(how="any", inplace=True)

    # create dictionary of dataframes only containing one variable each
    additional_columns = ["year", "month", "day", "decimal_month", "decimal_year"]
    ts_df_dict = {
        var: deepcopy(ts_df_new[[var] + additional_columns]) for var in variable_names
    }

    # remove NaNs
    # for k in ts_df_dict.keys():
    #     ts_df_dict[k].dropna(how="any", inplace=True)
    return ts_df_dict

def calc_ts_stats(ts_df_dict, var_sigfig_dict):
    # calculate and display monthly and annual statistics
    ts_stats = {}

    for k, v in ts_df_dict.items():
        ts_monthly_stats = round(v.groupby('month').agg({k: ['mean', 'std', 'count']}), var_sigfig_dict[k])
        ts_annual_stats = round(v.groupby('year').agg({k: ['mean', 'std', 'count']}), var_sigfig_dict[k])
        ts_stats[k] = {'monthly': ts_monthly_stats, 'annual': ts_annual_stats}

    return ts_stats

def calc_trends(ts_df_dict, var_unc_dict, var_sigfig_dict):
    dp_str_dict = {
        k: "%." + str(var_sigfig_dict[k]) + "f" for k in var_sigfig_dict.keys()
    }
    var_trends = {}
    for k, v in ts_df_dict.items():
        # fit linear model
        decimal_year = np.reshape(v['decimal_year'].to_numpy(), (-1,1))
        decimal_year = sm.add_constant(decimal_year)
        ts_variable = v[k].to_numpy()
        ts_variable = np.reshape(ts_variable, (-1,1))
        model = sm.OLS(ts_variable, decimal_year).fit(cov_type='HAC', cov_kwds={'maxlags':1})
        
        # calculate trend
        trend_to_remove = model.predict(decimal_year)

        # temporarily remove trend
        detrended = [ts_variable[i]-trend_to_remove[i] for i in range(0, len(decimal_year))]

        # extract slope and error values from OLS results and print results
        trend_to_remove_slope = dp_str_dict[k] % model.params[1]
        trend_to_remove_slope_error = dp_str_dict[k] % model.bse[1]
        var_trends[k] = {'model': model, 'detrended': detrended, 'slope_str': trend_to_remove_slope, 'slope_error_str': trend_to_remove_slope_error}

        #create new dataframe with de-trended values
    for k, v in var_trends.items():
        detrended = np.round(v['detrended'], var_sigfig_dict[k])
        detrended_df = pd.DataFrame(data=detrended, index=ts_df_dict[k].index, columns=['detrended_variable'])
        detrended_df['month'] = detrended_df.index.month
        detrended_df['year'] = detrended_df.index.year
        v['detrended'] = detrended_df

        # climatological monthly mean of de-trended values
    for k, v in var_trends.items():
        ts_month = v['detrended'].groupby('month')
        climatological_df = round(ts_month.agg({'detrended_variable': ['mean']}), var_sigfig_dict[k])

        # climatological annual mean of de-trended values
        annual_mean = np.mean(climatological_df['detrended_variable']['mean'])

        # monthly seasonal adjustment 
        climatological_df['monthly_adj'] = np.round(climatological_df['detrended_variable']['mean'].values - annual_mean, var_sigfig_dict[k])

        # seasonal amplitude and interannual variability for display in summary report
        season_max = climatological_df.detrended_variable['mean'].max()
        season_min = climatological_df.detrended_variable['mean'].min()
        seasonal_amplitude = round(season_max - season_min, var_sigfig_dict[k])

        annual_means = v['detrended'].groupby('year') 
        annual_means_df = annual_means.agg({'detrended_variable':['mean']})
        IAV = round(np.mean(abs(annual_means_df.detrended_variable['mean'])), var_sigfig_dict[k])
        v['climatological'] = climatological_df
        v['seasonal_amplitude'] = seasonal_amplitude
        v['annual_means'] = annual_means

    for k, v in var_trends.items():
        ts_year_month = ts_df_dict[k].groupby(['year','month'])

        ts_mean = round(ts_year_month.agg({k: ['mean']}), var_sigfig_dict[k])
        ts_mean[k,'datetime_mean'] = [datetime(i[0],i[1],15,0,0,0) for i in ts_mean.index]

        adj_mean = []
        for i in ts_mean.index:
            temp = ts_mean[k]['mean'][i] - v['climatological']['monthly_adj'][i[1]]
            adj_mean.append(temp)

        # create time series of de-seasoned monthly means (variable name = adj_mean)
        ts_mean[k, 'adj_mean'] = adj_mean
        v['ts_mean'] = ts_mean

    for k, v in var_trends.items():
        ts_mean = v['ts_mean']
        ts_mean[k,'year'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).year
        ts_mean[k,'month'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).month
        ts_mean[k,'day'] = pd.DatetimeIndex(ts_mean[k]['datetime_mean']).day
        ts_mean[k,'decimal_month'] = decimal_month(ts_mean[k]['year'], ts_mean[k]['month'], ts_mean[k]['day'])
        ts_mean[k, 'decimal_year'] = ts_mean[k]['year'] + (ts_mean[k]['decimal_month'] - 1) / 12
        v['ts_mean'] = ts_mean

    wls_model_dict = {}
    for k, v in var_trends.items():
        ts_mean = v['ts_mean']
        decimal_year_deseasoned = np.reshape(ts_mean[k]['decimal_year'].values, (-1, 1))
        min_year = np.amin(decimal_year_deseasoned)
        decimal_year_zero = decimal_year_deseasoned-min_year
        decimal_year_zero = sm.add_constant(decimal_year_zero)
        ts_variable_deseasoned = ts_mean[k]['adj_mean'].values
        ts_variable_deseasoned = np.reshape(ts_variable_deseasoned,(-1, 1))

        # Weights are based on user input uncertainty
        weights = var_unc_dict[k] * np.ones(len(ts_variable_deseasoned))

        # fit linear model
        weights = var_unc_dict[k]  # uncertainties
        model = sm.WLS(ts_variable_deseasoned,decimal_year_zero, weights).fit(cov_type='HAC',cov_kwds={'maxlags':1})
        wls_model_dict[k] = {'model': model, 'decimal_year_zero': decimal_year_zero, 'ts_variable_deseasoned': ts_variable_deseasoned}

    for k, v in wls_model_dict.items():
        model = v['model']
        trend = model.predict(v['decimal_year_zero'])

        # extract slope and error values from WLS results and print results
        slope_str = dp_str_dict[k]
        slope = slope_str % model.params[1]
        slope_error = slope_str % model.bse[1]
        wls_model_dict[k]['trend'] = trend
        wls_model_dict[k]['slope_str'] = slope
        wls_model_dict[k]['slope_err_str'] = slope_error

    return var_trends, wls_model_dict

def calc_TDT(var_trends, wls_model_dict, var_sigfig_dict):
    dp_str_dict = {
        k: "%." + str(var_sigfig_dict[k]) + "f" for k in var_sigfig_dict.keys()
    }
    TDTi_dict = {}  # time of detection
    # autocorrelation at lag 1 of the time series noise
    for k, v in wls_model_dict.items():
        ts_variable_deseasoned = v["ts_variable_deseasoned"]
        decimal_year_zero = v["decimal_year_zero"]
        autocorr = sm.tsa.stattools.acf(ts_variable_deseasoned, fft=False, nlags=1)[1:]
        ts_mean = var_trends[k]["ts_mean"]

        # standard deviation of detrended monthly anomalies
        model = v["model"]
        trend_to_remove_TDT = model.predict(decimal_year_zero)
        detrended_TDT = [
            ts_variable_deseasoned[i] - trend_to_remove_TDT[i]
            for i in range(0, len(ts_mean[k]["datetime_mean"]))
        ]
        std_dev = np.std(detrended_TDT)

        # time of detection
        TDTi = np.round(
            (
                ((3.3 * std_dev) / (abs(model.params[1:])))
                * (np.sqrt(((1 + autocorr) / (1 - autocorr))))
            )
            ** (2 / 3),
            1,
        )
        ts_length = round(np.max(decimal_year_zero[:, 1]), 1)

        # uncertainties of time of detection due to unknown variance and autocorrelation
        uncert_factor = (4 / (3 * np.sqrt(len(ts_variable_deseasoned)))) * (
            np.sqrt(((1 + autocorr) / (1 - autocorr)))
        )
        upper_conf_intervali = TDTi * math.exp(uncert_factor)
        lower_conf_intervali = TDTi * math.exp(-uncert_factor)
        uncert_TDTi = np.round(
            ((upper_conf_intervali - TDTi) + (TDTi - lower_conf_intervali)) / 2, 1
        )

        dp_str = dp_str_dict[k]
        TDT = dp_str % TDTi[0]
        uncert_TDT = dp_str % uncert_TDTi[0]
        upper_conf_interval = dp_str % upper_conf_intervali[0]
        lower_conf_interval = dp_str % lower_conf_intervali[0]

        TDTi_dict[k] = {
            "TDTi": TDTi,
            "ts_length": ts_length,
            "TDT": TDT,
            "uncert_TDT": uncert_TDT,
            "upper_conf_interval": upper_conf_interval,
            "lower_conf_interval": lower_conf_interval,
        }
    return TDTi_dict

def calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict):
    # calculate and display monthly and annual statistics
    ts_stats = calc_ts_stats(ts_df_dict, var_sigfig_dict)
    var_trends, wls_model_dict  = calc_trends(ts_df_dict, var_unc_dict, var_sigfig_dict)
    TDTi_dict = calc_TDT(var_trends, wls_model_dict, var_sigfig_dict)

    return ts_stats, var_trends, wls_model_dict, TDTi_dict

def toU10(uz,z):
    z0 = z*np.exp(-3.7+1.165 * np.log(0.032*0.4**2*uz**2/9.81/z))
    u10 = np.log(10/z0)/np.log(z/z0)*uz
    return u10

def bin_counts(x,y,xlim,ylim,step):
    x_new = np.arange(xlim[0],xlim[1]+step/2,step)
    y_new = np.arange(ylim[0],ylim[1]+step/2,step)

    dens = np.full((len(y_new),len(x_new)),np.nan)
    for idxx in range(len(x_new)):
        for idxy in range(len(y_new)):
            n = np.sum((x>=x_new[idxx]-step/2) & (x<x_new[idxx]+step/2) & (y>=y_new[idxy]-step/2) & (y<y_new[idxy]+step/2))
            if n>0:
                dens[idxx,idxy] = np.copy(n)
    return dens,x_new,y_new

def get_gom_mask(lon,lat,file_topo='/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/etopo/ETOPO2v2c_f4.nc'):
    lat_msh,lon_msh = np.meshgrid(lat,lon,indexing='ij')
    # path_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/OISSTv2_highres/'
    # file_sst = path_sst + 'sst.day.mean.'+ str(yr_min) + '.v2.nc'
    # ds_i = xr.open_dataset(file_sst)
    # idx_lon_sst = np.where((ds_i.lon.values>=lon_min+360) & (ds_i.lon.values<=lon_max+360))[0]
    # idx_lat_sst = np.where((ds_i.lat.values>=lat_min) & (ds_i.lat.values<=lat_max))[0]
    # lon_sst = ds_i.lon[idx_lon_sst].values-360
    # lat_sst = ds_i.lat[idx_lat_sst].values
    # sst_i = np.nanmean(ds_i.sst.interp(lon=lon+360,lat=lat).values,axis=0)
    mask_gom = np.ones(lat_msh.shape)

    x1 = -83
    y1 = 31
    x2 = -81
    y2 =25.5
    a = (y2-y1)/(x2-x1)
    b = y2-a*x2
    mask_gom[(a*lon_msh+b)<lat_msh] = np.nan

    # file_topo = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/etopo/ETOPO2v2c_f4.nc'
    ds_topo = xr.open_dataset(file_topo)
    depth = ds_topo.z.interp(x=lon,y=lat).values

    x1 = -87.0
    y1 = 21
    x2 = -80.5
    y2 =25
    a = (y2-y1)/(x2-x1)
    b = y2-a*x2

    # mask_gom[np.isnan(sst_i)] = np.nan
    mask_gom[(a*lon_msh+b)>lat_msh] = np.nan
    mask_gom[(lon_msh<=-89.75) & (lat_msh>=29.5)] = np.nan
    mask_gom[depth>=0] = np.nan

    b_txs_las = -93
    b_las_wfs = -86
    b_cb_ys = -90
    b_nsgom = 24
    depth_thres = -200

    mask_ngom = mask_gom.copy()
    mask_ngom[(lat_msh<b_nsgom)] = np.nan

    mask_sgom = mask_gom.copy()
    mask_sgom[(lat_msh>=b_nsgom)] = np.nan

    mask_ncoastal = mask_ngom.copy()
    mask_ncoastal[(depth<depth_thres)] = np.nan

    mask_txs = mask_ncoastal.copy()
    mask_txs[lon_msh>=b_txs_las] = np.nan

    mask_las = mask_ncoastal.copy()
    mask_las[(lon_msh<b_txs_las) | (lon_msh>=b_las_wfs)] = np.nan

    mask_wfs = mask_ncoastal.copy()
    mask_wfs[(lon_msh<b_las_wfs)] = np.nan

    mask_nopen = mask_ngom.copy()
    mask_nopen[(depth>=depth_thres)] = np.nan

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

    mask_cb = mask_sgom.copy()
    mask_cb[lon_msh>=b_cb_ys] = np.nan
    mask_ys = mask_sgom.copy()
    mask_ys[lon_msh<b_cb_ys] = np.nan

    mask = dict({'gom':mask_gom,
                'ngom':mask_ngom,
                'sgom':mask_sgom,
                'ncoastal':mask_ncoastal,
                'nopen':mask_nopen,
                'txs':mask_txs,
                'las':mask_las,
                'wfs':mask_wfs,
                'nopenw':mask_nopenw,
                'nopenm':mask_nopenm,
                'nopene':mask_nopene,
                'nopenm1':mask_nopenm1,
                'nopenm2':mask_nopenm2,
                'cb':mask_cb,
                'ys':mask_ys,})
    return mask  
# %%
lon_min = -100
lon_max = -78
lat_min = 17
lat_max = 31
yr_min = 2002
yr_max = 2020
time_min = np.datetime64('2002-01-01')
time_max = np.datetime64('2020-12-31')
#%%
# National Data Bouy Center
# Read bouy information
bouyinfo = pd.read_csv('/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/NDBC/nGOM/GOMbuoyinfo.csv')
lon_bouy = bouyinfo.lon.values
lat_bouy = bouyinfo.lat.values
name_bouy = bouyinfo.Mooring.values

yr_start_bouy = max(bouyinfo.StartTime)
yr_end_bouy = min(bouyinfo.EndTime)
height_bouy = bouyinfo.AnemometerHeight.values

# Read NCEP
file_wspd_ncep2 = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/NCEP-DOE Reanalysis 2/wspd.10m.mon.mean.nc'
ds_wspd_ncep2 = xr.open_dataset(file_wspd_ncep2)
idxlonncep2 = np.where((ds_wspd_ncep2.lon>=lon_min+360) & (ds_wspd_ncep2.lon<=lon_max+360))[0]
idxlatncep2 = np.where((ds_wspd_ncep2.lat>=lat_min) & (ds_wspd_ncep2.lat<=lat_max))[0]

# Read ERA5
file_era5 = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/ERA5/era5_monthly_u10_global.nc';
ds_era5 = xr.open_dataset(file_era5)

file_ccmp = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y2003/M01/CCMP_Wind_Analysis_200301_V02.0_L3.5_RSS.nc'
ds_ccmp = xr.open_dataset(file_ccmp)

idx_lon_ncep2_boy = np.full_like(lon_bouy,np.nan)
idx_lon_ccmp_boy = np.full_like(lon_bouy,np.nan)
idx_lon_era5_boy = np.full_like(lon_bouy,np.nan)

idx_lat_ncep2_boy = np.full_like(lat_bouy,np.nan)
idx_lat_ccmp_boy = np.full_like(lat_bouy,np.nan)
idx_lat_era5_boy = np.full_like(lat_bouy,np.nan)

for idxb in range(len(lon_bouy)):
    idx_lon_ncep2_boy[idxb] = np.where(np.abs(ds_wspd_ncep2.lon.values - 360 - lon_bouy[idxb]) == np.min(np.abs(ds_wspd_ncep2.lon.values - 360 - lon_bouy[idxb])))[0][0]
    idx_lat_ncep2_boy[idxb] = np.where(np.abs(ds_wspd_ncep2.lat.values - lat_bouy[idxb]) == np.min(np.abs(ds_wspd_ncep2.lat.values - lat_bouy[idxb])))[0][0]

    idx_lon_era5_boy[idxb] = np.where(np.abs(ds_era5.longitude.values - 360 - lon_bouy[idxb]) == np.min(np.abs(ds_era5.longitude.values - 360 - lon_bouy[idxb])))[0][0]
    idx_lat_era5_boy[idxb] = np.where(np.abs(ds_era5.latitude.values - lat_bouy[idxb]) == np.min(np.abs(ds_era5.latitude.values - lat_bouy[idxb])))[0][0]

    idx_lon_ccmp_boy[idxb] = np.where(np.abs(ds_ccmp.longitude.values - 360 - lon_bouy[idxb]) == np.min(np.abs(ds_ccmp.longitude.values - 360 - lon_bouy[idxb])))[0][0]
    idx_lat_ccmp_boy[idxb] = np.where(np.abs(ds_ccmp.latitude.values - lat_bouy[idxb]) == np.min(np.abs(ds_ccmp.latitude.values - lat_bouy[idxb])))[0][0]
#%% Section 4.1 Wind speeds and xCO2air
df_wspd = pd.DataFrame({'year':[],
                        'month':[],
                        'bouy':[],
                        'bouy_lon':[],
                        'bouy_lat':[],
                        'bouy_wspd':[],
                        'ccmp':[],
                        'era5':[],
                        'ncep2':[],})

for idx_name in np.arange(len(lon_bouy)):
    df_b = pd.DataFrame({'year':[],
                           'month':[],
                           'day':[],
                           'wdir':[],
                           'wspd':[]})

    for idx_yr in range(yr_start_bouy,yr_end_bouy+1):
        file_b = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/NDBC/nGOM/' + name_bouy[idx_name] + '/' + name_bouy[idx_name] + 'c' + str(idx_yr) + '.txt'
        if os.path.exists(file_b):
            data_i = pd.read_csv(file_b,sep='\s+')
            ii_drop = np.array([])
            for ii in range(data_i.shape[0]):
                try:
                    np.float32(data_i.iloc[ii,:].values)
                except:
                    ii_drop = np.append(ii_drop,int(ii))
            data_i = data_i.drop(index=ii_drop)
            data_i.index=np.arange(data_i.shape[0])
            df_i = pd.DataFrame({'year':np.float32(data_i.iloc[:,0].values),
                        'month':np.float32(data_i.iloc[:,1].values),
                        'day':np.float32(data_i.iloc[:,2].values),
                        'wdir':np.float32(data_i.iloc[:,5].values),
                        'wspd':np.float32(data_i.iloc[:,6].values)})
            df_b = df_b.append(df_i,ignore_index=True)
    df_b.index = np.arange(df_b.shape[0])
    df_b_m = df_b.groupby(['year','month']).mean()
    df_b_m.wspd = toU10(df_b_m.wspd.values,height_bouy[idx_name])
    
    bouy_label_i = np.full(df_b_m.shape[0],name_bouy[idx_name])
    bouy_lon_i = np.full(df_b_m.shape[0],lon_bouy[idx_name])
    bouy_lat_i = np.full(df_b_m.shape[0],lat_bouy[idx_name])

    year_i = np.full(df_b_m.shape[0],np.nan)
    month_i = np.full(df_b_m.shape[0],np.nan)

    wspd_bouy_i = np.full(df_b_m.shape[0],np.nan)
    wspd_ccmp_i = np.full(df_b_m.shape[0],np.nan)
    wspd_era5_i = np.full(df_b_m.shape[0],np.nan)
    wspd_ncep2_i = np.full(df_b_m.shape[0],np.nan)

    for idxrow in range(df_b_m.shape[0]):
        year_i[idxrow] = np.copy(int(df_b_m.index[idxrow][0]))
        month_i[idxrow] = np.copy(int(df_b_m.index[idxrow][1]))

        wspd_bouy_i[idxrow] = df_b_m.wspd.values[idxrow]

        idx_t_ncep2 = int((df_b_m.index[idxrow][0] - 1979)*12 + df_b_m.index[idxrow][1])
        wspd_ncep2_i[idxrow] = ds_wspd_ncep2.wspd[idx_t_ncep2,0,int(idx_lat_ncep2_boy[idx_name]),int(idx_lon_ncep2_boy[idx_name])].values

        idx_t_era5 = int((df_b_m.index[idxrow][0] - 1979)*12 + df_b_m.index[idxrow][1])
        wspd_era5_i[idxrow] = ds_era5.si10[idx_t_era5,0,int(idx_lat_era5_boy[idx_name]),int(idx_lon_era5_boy[idx_name])].values

        file_ccmp = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y'+ str(int(df_b_m.index[idxrow][0]))+'/M'+str(int(df_b_m.index[idxrow][1])).zfill(2)+'/CCMP_Wind_Analysis_'+ str(int(df_b_m.index[idxrow][0]))+str(int(df_b_m.index[idxrow][1])).zfill(2)+'_V02.0_L3.5_RSS.nc'
        ds_ccmp = xr.open_dataset(file_ccmp)
        wspd_ccmp_i[idxrow] = ds_ccmp.wspd[0,int(idx_lat_ccmp_boy[idx_name]),int(idx_lon_ccmp_boy[idx_name])].values

    df_re_i = pd.DataFrame({'year':year_i,
                            'month':month_i,
                            'bouy':bouy_label_i,
                            'bouy_lon':bouy_lon_i,
                            'bouy_lat':bouy_lat_i,
                            'bouy_wspd':df_b_m.wspd.values,
                            'ccmp':wspd_ccmp_i,
                            'era5':wspd_era5_i,
                            'ncep2':wspd_ncep2_i,})
    df_wspd = df_wspd.append(df_re_i)

# %%
from statsmodels.tools.eval_measures import rmse
ylabels = ['CCMP U$_{10}$ (m/s)','ERA5 U$_{10}$ (m/s)','NCEP2 U$_{10}$ (m/s)']
titles = ['(a) CCMP & Bouy','(b) ERA5 & Bouy','(c) NCEP2 & Bouy']
fig = plt.figure(figsize=(11.5,5))
axgr = AxesGrid(fig, 111, 
                nrows_ncols=(1,3),
                axes_pad=0.7,
                cbar_location='bottom',
                cbar_mode='single',
                cbar_pad=0.3,
                cbar_size='2.5%',
                label_mode='')  # note the empty label_mode
for i, ax in enumerate(axgr):
    x = df_wspd.bouy_wspd.values
    y = df_wspd.iloc[:,i+6].values

    lm = sm.OLS(y,sm.add_constant(x)).fit()

    dens, x_new, y_new = bin_counts(x,y,[0,11],[0,11],0.2)
    p = ax.pcolor(x_new,y_new,dens.T,clim=[0,10],cmap=plt.get_cmap('RdYlBu_r',20))
    # ax.scatter(x,y,s=5)
    ax.plot([0,11],[0,11],linewidth=2,c='r')
    
    y_est = lm.predict(sm.add_constant(x))
    rse = rmse(y,y_est)
    ax.plot(x,y_est,linewidth=1.5,c='b')

    ax.text(6.5,1.5,'R$^2$ = '+'{:.2f}'.format(lm.rsquared))
    ax.text(6.5,0.9,'p < 0.0001')
    ax.text(6.5,0.3,'rmse = '+'{:.2f}'.format(rse) + ' m/s')

    ax.set_xlabel('Bouy  U$_{10}$ (m/s)')
    ax.set_ylabel(ylabels[i])
    ax.set_xlim([0,11])
    ax.set_ylim([0,11])

    ax.text(-0.2, 11.3,titles[i],fontdict={'size':10,'weight':'bold'})

axgr.cbar_axes[0].colorbar(p,extend='max',label='Months (bin interval = 0.2 m/s)')
plt.tight_layout()
plt.savefig('figs/Fig10_Discussion_U10Comparison.jpg',dpi=300)
plt.show()
# %% xCO2air
#%%
file_socat = 'data/socat_bin025_GOM.nc'
ds_socat = xr.open_dataset(file_socat)
idx_lon_socat = np.where((ds_socat.lon.values>=lon_min) & (ds_socat.lon.values<=lon_max))[0]
idx_lat_socat = np.where((ds_socat.lat.values>=lat_min) & (ds_socat.lat.values<=lat_max))[0]
idx_time_socat = np.where((ds_socat.time.values>=time_min) & (ds_socat.time.values<=time_max))[0]
lon_socat = ds_socat.lon[idx_lon_socat].values
lat_socat = ds_socat.lat[idx_lat_socat].values
time = pd.to_datetime(ds_socat['time'][idx_time_socat].values)
year_socat = time.year
month_socat = time.month
day_socat = time.day
# fco2sw = ds_socat.fCO2rec[idx_time_socat,idx_lat_socat,idx_lon_socat].values
# sst = ds_socat.SST[idx_time_socat,idx_lat_socat,idx_lon_socat].values
# sss = ds_socat.sal[idx_time_socat,idx_lat_socat,idx_lon_socat].values
# xco2air_gv = ds_socat.GVCO2[idx_time_socat,idx_lat_socat,idx_lon_socat].values

# file_co2air = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/NOAA_MBL_Reference/MBLCO2_24N31N.csv'
# df_co2air = pd.read_csv(file_co2air,header=74,sep='\s+')
# df_xco2air_m = df_co2air.groupby(['year','month']).mean()
# xco2air = df_xco2air_m.value.values[(df_xco2air_m.decimal_date>=yr_min)&(df_xco2air_m.decimal_date<=yr_max)]
# xco2air_err = df_xco2air_m.uncertainty.values[(df_xco2air_m.decimal_date>=yr_min)&(df_xco2air_m.decimal_date<yr_max)] #0.07%
# xco2air_3d = np.repeat(np.repeat(np.reshape(xco2air,(len(xco2air),1,1)),len(lat_socat),axis=1),len(lon_socat),axis=2)
#%%
file_gv_info = pd.read_excel('/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CO2_air/obspack_co2_1_GLOBALVIEWplus_v8.0_2022-08-27_txt/GOM_stations.xlsx')
file_gv_info = file_gv_info.drop(index=[0,1])
file_gv_info.index = range(file_gv_info.shape[0])
df_gv_all = pd.DataFrame({'year':[],
                          'month':[],
                          'value':[]})
for idxf in range(file_gv_info.shape[0]):
    df_gv_i = pd.read_csv(file_gv_info.path[idxf],header=file_gv_info.header[idxf],sep='\s+')
    df_gv_section_i = df_gv_i[['year','month','value']]
    idxrow = np.where((df_gv_section_i['year'].values>=yr_min) & (df_gv_section_i['year'].values<=yr_max))[0]
    df_gv_all = df_gv_all.append(df_gv_section_i.iloc[idxrow,:])
    # print(df_gv_i.head())
df_gv_all_monavg = df_gv_all.groupby(['year','month']).mean()
#%% GV daily
files = np.sort(glob('/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CO2_air/obspack_co2_1_GLOBALVIEWplus_v8.0_2022-08-27_daily/data/daily/*.nc'))
time_gv_daily = pd.to_datetime([])
xco2air_gvdaily = np.array([])
for file_i in files:
    if (np.float32(file_i[-11:-7]) >=yr_min) & (np.float32(file_i[-11:-7]) <=yr_max):
        ds_gvdaily = xr.open_dataset(file_i)
        lon_i = ds_gvdaily.longitude.values
        lat_i = ds_gvdaily.latitude.values
        idxloc = np.where((lon_i>=-100) & (lon_i<=-81) & (lat_i>=18) & (lat_i<=32))[0]
        if len(idxloc)>1:
            time_gv_daily = np.append(time_gv_daily,pd.to_datetime(file_i[-11:-3]))
            xco2air_gvdaily = np.append(xco2air_gvdaily,np.nanmean(ds_gvdaily.value[idxloc].values))

time_gv_daily = pd.to_datetime(time_gv_daily)
df_gvdaily = pd.DataFrame({'year':time_gv_daily.year,
                           'month':time_gv_daily.month,
                           'time':time_gv_daily,
                           'value':xco2air_gvdaily})
df_gvdaily_monavg = df_gvdaily.groupby(['year','month']).mean()
time_gvdaily_monavg = np.array([np.datetime64(str(df_gvdaily_monavg.index[k][0])+'-'+str(df_gvdaily_monavg.index[k][1]).zfill(2)+'-15') for k in range(df_gvdaily_monavg.shape[0])])

#%%
file_maunaloa = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CO2_air/Mauna_Loa.csv'
df_maunaloa = pd.read_csv(file_maunaloa,header=58,sep='\s+')
idxrow = np.where((df_maunaloa.year.values>=yr_min) & (df_maunaloa.year.values<=yr_max))[0]
xco2air_maunaloa = df_maunaloa.mon_avg.iloc[idxrow].values
#%% MBL
file_mbl = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CO2_air/MBLReferenceCO2/MBLCO2_18N31N_zonnal_avg.csv'
df_mbl = pd.read_csv(file_mbl,header=74,sep='\s+')
idxrow = np.where((df_mbl.year.values>=yr_min) & (df_mbl.year.values<=yr_max))[0]

df_mbl_monavg = df_mbl.iloc[idxrow,:].groupby(['year','month']).mean()
#%% Carbon Tracker
xco2air_ct = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
for idxyr in range(yr_min,yr_max+1):
    for idxmon in range(1,13):
        file_ct = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CarbonTracker/CT2022GB/CT2022.molefrac_glb3x2_'+str(idxyr)+'-'+str(idxmon).zfill(2)+'.nc'
        ds_ct_i = xr.open_dataset(file_ct)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        xco2air_ct[idx_i,:,:] = ds_ct_i.co2[0,0,:,:].interp(longitude=lon_socat,latitude=lat_socat).values
# %%
mask_socat = get_gom_mask(lon_socat,lat_socat)
ts_gv = df_gv_all_monavg.value.values * 10**6
ts_ct = np.nanmean(xco2air_ct*mask_socat['gom'],axis=(1,2))
ts_maunaloa = xco2air_maunaloa
ts_mbl = df_mbl_monavg.value.values
ts_gvdaily = df_gvdaily_monavg.value.values*10**6

#%%
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
# %%
fig = plt.figure(figsize=(13.5,4))
plt.subplots_adjust(left=0.05,
                    bottom=0.1,
                    right=0.95,
                    top=0.9,
                    wspace=0.1,
                    hspace=0.1)
plt.tight_layout()
ax = plt.subplot(1,2,1,projection=projection)
ax.coastlines()
ax.add_feature(cfeature.LAND,facecolor='grey',alpha=0.3)
ax.add_feature(cfeature.RIVERS)
ax.set_xticks(np.linspace(-98, -82, 3), crs=projection)
ax.set_yticks(np.linspace(20, 30, 3), crs=projection)
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.set_xlim([-98.5,-80])
ax.set_ylim([18,31])
p1 = ax.contourf(lon_socat, lat_socat, np.nanmean(xco2air_ct,axis=0), transform=projection,levels = np.arange(390,404.1,1),cmap=plt.get_cmap('RdYlBu_r',14),extend='both')
s1 = ax.scatter(file_gv_info.longitude.values,file_gv_info.latitude.values,s = 100, c= 'blue',marker='o')
plt.colorbar(p1,extend='both',label='xCO$_{2air}$ (ppm)')
for ii in range(3):
    if ii == 0:
        ax.text(file_gv_info.longitude.values[ii]-1.5,file_gv_info.latitude.values[ii]+0.5,file_gv_info['Site code'][ii],fontdict={'size':12},color='b')
    elif ii == 1:
        ax.text(file_gv_info.longitude.values[ii]+0.5,file_gv_info.latitude.values[ii]+0.5,file_gv_info['Site code'][ii],fontdict={'size':12},color='b')
    else:
        ax.text(file_gv_info.longitude.values[ii]-0.4,file_gv_info.latitude.values[ii]-1.2,file_gv_info['Site code'][ii],fontdict={'size':12},color='b')

ax.text(-99,31.5,'(d) CT2022 xCO$_{2air}$ & GV+ stations',fontdict={'size':14,'weight':'bold'})

plt.subplot(1,2,2)
p1 = plt.plot(time,ts_gv,label='GLOBALVIEWplus (GV+) ObsPacks')
p3 = plt.plot(time,ts_maunaloa,label='Mauna Loa (Keeling Curve)')
p4 = plt.plot(time,ts_mbl,label='CO2 MBL Reference')
p2 = plt.plot(time,ts_ct,label='CarbonTracker CT2022')
# p5 = plt.plot(time_gvdaily_monavg,ts_gvdaily,label='GLOBALVIEWplus (GV+) ObsPacks daily avg')
plt.xlim([time[0],time[-1]])
plt.ylim([365,425])
# plt.xticks(np.arange(2002,2021))
# plt.text(np.datetime64('2010-01-01'),378,'Δ(GV - CT2022) = '+'{:+.1f}'.format(np.nanmean(ts_gv - ts_ct)) + ' ± ' + '{:.1f}'.format(np.nanstd(ts_gv - ts_ct)) + ' ppm')
# plt.text(np.datetime64('2010-01-01'),374,'Δ(Mauna Loa - CT2022) = '+'{:+.1f}'.format(np.nanmean(ts_maunaloa - ts_ct)) + ' ± ' + '{:.1f}'.format(np.nanstd(ts_maunaloa - ts_ct)) + ' ppm')
# plt.text(np.datetime64('2010-01-01'),370,'Δ(MBL - CT2022) = '+'{:+.1f}'.format(np.nanmean(ts_mbl - ts_ct)) + ' ± ' + '{:.1f}'.format(np.nanstd(ts_mbl - ts_ct)) + ' ppm')
ts_all = np.array([ts_gv,ts_maunaloa,ts_mbl,ts_ct])
plt.text(np.datetime64('2014-01-01'),374,'σ = '+ '{:.2f}'.format(np.nanmean(np.nanstd(ts_all,axis=0)))+' ppm',fontdict={'size':14})

plt.text(np.datetime64('2001-09-01'),427,'(e) Time series of 4 xCO$_{2air}$ data products',fontdict={'size':14,'weight':'bold'})
plt.legend()

plt.grid('on',alpha=0.2)
plt.ylabel('xCO$_{2air}$ (ppm)')
plt.savefig('figs/Fig10_Discussion_xCO2airComparison.jpg',dpi=300)
# %%
