#%%
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

# 3rd party libraries
import statsmodels.api as sm
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
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
#%%
def co_K0_Weiss(sst, sss):
    ###############################################
    # pco2 = CO_XCO2TOPCO2(sst, sss)
    # A function calculated the CO2 solubility with SST (C) and SSS
    # Uncertainty is 2# according to Weiss (1974).
    #
    # Input:
    #   sst: sea surface temperature
    #   sss: sea surface salinity (degree Celcius)
    # Output:
    #   K0: CO2 solubility, unit: mol/l/atm
    #
    # Algorithm:
    # ln_K0 = A1 + A2.*(100./sst) + A3 *log(sst/100) + sss.* (B1+ B2*(sst/100)+ B3*(sst/100).^2);
    # K0 = exp(ln_K0);
    #
    # References:
    # Weiss, R. F. (1974). Carbon dioxide in water and seawater: the solubility of a non-ideal gas. Marine Chemistry, 2(3), 203–215. https://doi.org/10.1016/0304-4203(74)90015-2
    #
    # Zelun Wu
    # Ph.D. student.
    # University of Delaware & Xiamen University
    # zelunwu@outlook.com
    # zelunwu.github.io
    # #############################################

    sss = np.array(sss)
    sst = np.array(sst)
    sst = np.array(sst + 273.15)  # transfer to Kelvin degree
    A1, A2, A3 = -58.0931, 90.5069, 22.294
    B1, B2, B3 = 0.027766, -0.025888, 0.0050578
    ln_K0 = (
        A1
        + A2 * (100 / sst)
        + A3 * np.log(sst / 100)
        + sss * (B1 + B2 * (sst / 100) + B3 * (sst / 100) ** 2)
    )
    K0 = np.exp(ln_K0)
    return K0

def co_gas_transfer_velocity(sst, wspd, c=0.251, unit="cmhr"):
    # kt = co_gas_transfer_velocity(sst,wspd,c,unit)
    # Calculate the gas transfer velocity from sst, sss, and wind speed
    # Input:
    #     sst: sea surface temperature
    #     wspd: wind speed
    #     c: coefficient, default is 0.251 (Wanninkhof, 2014)
    #     unit: 1, or 2. 1 represents unit for k is hour/cm; 2 represents m/yr. default is 1.
    #
    # Output:
    #     kt: gas transfer velocity
    #
    # Uncertainty is 20# on the global average, but should be calibrated with the mooring measured wind speed in the
    # regional ocean.
    #
    # Algorithm:
    # kt = c * wpsd^2 * (Sc/660)^-0.5
    # Sc is the Schmit numer, which is a function of SST:
    # Sc = A + B*(sst) + C*(sst.^2) + D*(sst.^3) + E*(sst.^4);
    # A = 2116.8; B = -136.25; C = 4.7353; D = -0.092307; E = 0.0007555;
    #
    # References:
    # Wanninkhof, R. (2014). Relationship between wind speed and gas exchange over the ocean revisited. Limnology and Oceanography: Methods, 12(6), 351–362. https://doi.org/10.4319/lom.2014.12.351
    #
    # Author:
    # Zelun Wu
    # Ph.D. student.
    # University of Delaware & Xiamen University
    # zelunwu@outlook.com
    # zelunwu.github.io
    ##
    sst = np.array(sst)
    wspd = np.array(wspd)

    A, B, C, D, E = 2116.8, -136.25, 4.7353, -0.092307, 0.0007555
    Sc = (
        A + B * (sst) + C * (sst**2) + D * (sst**3) + E * (sst**4)
    )  # Jähne et al. (1987), Wanninkhof 2014
    kt = c * wspd**2 * ((Sc / 660) ** (-0.5))  # unit: cm/hour
    if not (unit == "cmhr"):
        kt = kt * (24 * 365 / 100)  # unit: m/yr
    return kt


def co_co2flux(pCO2sea, pCO2air, sst, sss, wspd):
    pCO2sea = np.array(pCO2sea)
    pCO2air = np.array(pCO2air)
    sst = np.array(sst)
    sss = np.array(sss)
    wspd = np.array(wspd)

    kt = co_gas_transfer_velocity(sst, wspd, 0.251)
    K0 = co_K0_Weiss(sst, sss)
    dpco2 = pCO2sea - pCO2air
    # Because kt is in cm/hour, pCO2 is in µatm, flux in most case is in mol C/m2/yr
    F = kt * K0 * dpco2 * (24 * 365 / 100000)
    return F

def co_xco2topco2(xco2, sst, sss, p=101325):
    # pco2 = CO_XCO2TOPCO2(xco2, sst, sss)
    # A function converted atmospheric mole fraction of CO2 (xCO2) to atmospheric pCO2
    #
    # Algorithm:
    # fco2 = xco2 * (1-pw);
    # pw = exp()...
    #
    # References:
    #
    #
    # Author:
    # Zelun Wu
    # Ph.D. student.
    # University of Delaware & Xiamen University
    # zelunwu@outlook.com
    # zelunwu.github.io
    ##

    xco2 = np.array(xco2)
    sst = np.array(sst)
    sss = np.array(sss)

    C1 = 24.4543
    C2 = -67.4509 * (100 / (sst + 273.15))
    C3 = -4.8489 * np.log((sst + 273.15) / 100)
    C4 = -0.000544 * sss
    pw = np.exp(C1 + C2 + C3 + C4)
    pco2 = xco2 * (p/101325 - pw)
    return pco2

def p2fCO2(pCO2,T=25, P=0, Patm=1):
    # Copyright (C) 2010  Héloïse Lavigne and Jean-Pierre Gattuso
    # with a most valuable contribution of Bernard Gentili <gentili@obs-vlfr.fr>
    # and valuable suggestions from Jean-Marie Epitalon <epitalon@lsce.saclay.cea.fr>
    #
    # This file is part of seacarb.
    #
    # Seacarb is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; either version 2 of the License, or any later version.
    #
    # Seacarb is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
    #
    # You should have received a copy of the GNU General Public License along with seacarb; if not, write to the Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
    #
    #
    #

    # Original "seacarb" f2pCO2 calculation:
    # B <- (-1636.75+12.0408*TK-0.0327957*(TK*TK)+0.0000316528*(TK*TK*TK))*1e-6
    # fCO2 <-  pCO2*(1/exp((1*100000)*(B+2*(57.7-0.118*TK)*1e-6)/(8.314*TK)))^(-1)
    # Above calculation:
    # - uses incorrect R (wrong units, incompatible with pressure in atm)
    # - neglects a term "x2" (see below)
    # - assumes pressure is always 1 atm (wrong for subsurface)
    tk = 273.15;           # [K] (for conversion [deg C] <-> [K])
    TK = T + tk;           # TK [K]; T[C]
    Phydro_atm = P / 101325  # convert hydrostatic pressure from Pa to atm (1.01325 bar / atm)
    Ptot = Patm + Phydro_atm  # total pressure (in atm) = atmospheric pressure + hydrostatic pressure

    # To compute fugcoeff, we need 3 other terms (B, Del, xc2) in addition to 3 others above (TK, Ptot, R)
    B   = -1636.75 + 12.0408*TK - 0.0327957*TK**2 + 0.0000316528*TK**3
    Del = 57.7-0.118*TK

    # "x2" term often neglected (assumed = 1) in applications of Weiss's (1974) equation 9
    # x2 = 1 - x1 = 1 - xCO2 (it is close to 1, but not quite)
    # Let's assume that xCO2 = pCO2. Resulting fugcoeff is identical to at least 8th digit after the decimal.
    xCO2approx = pCO2
    xc2 = (1 - xCO2approx*1e-6)**2 

    fugcoeff = np.exp( Ptot*(B + 2*xc2*Del)/(82.057*TK) )
    fCO2 = pCO2 * fugcoeff
    return fCO2
#%%
lon_min = -100
lon_max = -78
lat_min = 17
lat_max = 31
yr_min = 2002
yr_max = 2020
time_min = np.datetime64('2002-01-01')
time_max = np.datetime64('2020-12-31')
#%% Read SOCAT Coastal
file_socat = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/socat/SOCATv2022_qrtrdeg_gridded_coast_monthly.nc'
ds_socat = xr.open_dataset(file_socat)
idx_lon_socat = np.where((ds_socat.xlon.values>=lon_min) & (ds_socat.xlon.values<=lon_max))[0]
idx_lat_socat = np.where((ds_socat.ylat.values>=lat_min) & (ds_socat.ylat.values<=lat_max))[0]
idx_time_socat = np.where((ds_socat.tmnth.values>=time_min) & (ds_socat.tmnth.values<=time_max))[0]
lon_socat = ds_socat.xlon[idx_lon_socat].values
lat_socat = ds_socat.ylat[idx_lat_socat].values
time = pd.to_datetime(ds_socat['tmnth'][idx_time_socat].values)
year_socat = time.year
month_socat = time.month
day_socat = time.day
fco2sw1 = ds_socat.coast_fco2_ave_unwtd[idx_time_socat,idx_lat_socat,idx_lon_socat].values
sst1 = ds_socat.coast_sst_ave_weighted[idx_time_socat,idx_lat_socat,idx_lon_socat].values
sss1 = ds_socat.coast_salinity_ave_weighted[idx_time_socat,idx_lat_socat,idx_lon_socat].values

sst1[(fco2sw1<1) | (sst1<-4)| (sss1<0)] = np.nan
sss1[(fco2sw1<1) | (sst1<-4)| (sss1<0)] = np.nan
fco2sw1[(fco2sw1<1) | (sst1<-4)| (sss1<0)] = np.nan
sss1[(sss1>np.percentile(sss1[~np.isnan(sss1)],99.9))|(sss1<0.5)] = np.nan
sst1[(sst1>np.percentile(sst1[~np.isnan(sst1)],99.9))|(sst1<0.1)] = np.nan
fco2sw1[(fco2sw1>np.percentile(fco2sw1[~np.isnan(fco2sw1)],99.9))|(fco2sw1<np.percentile(fco2sw1[~np.isnan(fco2sw1)],0.1))] = np.nan

# results = sys(par1=fco2sw1,par2=1000,par1_type=5,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst)
# pco2sw = results['pCO2']
# pco2sw[pco2sw>2000] = np.nan

# sst_clim = np.nanmean(np.reshape(sst,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
# sss_clim = np.nanmean(np.reshape(sss,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
# fco2sw_clim = np.nanmean(np.reshape(fco2sw,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
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
fco2sw = ds_socat.fCO2rec[idx_time_socat,idx_lat_socat,idx_lon_socat].values
sst = ds_socat.SST[idx_time_socat,idx_lat_socat,idx_lon_socat].values
sss = ds_socat.sal[idx_time_socat,idx_lat_socat,idx_lon_socat].values
xco2air_gv = ds_socat.GVCO2[idx_time_socat,idx_lat_socat,idx_lon_socat].values

sst[(fco2sw<1) | (sst<-4)| (sss<0)] = np.nan
sss[(fco2sw<1) | (sst<-4)| (sss<0)] = np.nan
fco2sw[(fco2sw<1) | (sst<-4)| (sss<0)] = np.nan
xco2air_gv[xco2air_gv<1] = np.nan

sss[(sss>np.percentile(sss[~np.isnan(sss)],99.9))|(sss<0.5)] = np.nan
sst[(sst>np.percentile(sst[~np.isnan(sst)],99.9))|(sst<0.1)] = np.nan
fco2sw[(fco2sw>np.percentile(fco2sw[~np.isnan(fco2sw)],99.9))|(fco2sw<np.percentile(fco2sw[~np.isnan(fco2sw)],0.1))] = np.nan

results = sys(par1=fco2sw,par2=1000,par1_type=5,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst)
pco2sw = results['pCO2']
pco2sw[pco2sw>2000] = np.nan

sst_clim = np.nanmean(np.reshape(sst,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
sss_clim = np.nanmean(np.reshape(sss,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
fco2sw_clim = np.nanmean(np.reshape(fco2sw,(int(sst.shape[0]/12),12,sst.shape[1],sst.shape[2])),axis=0)
#%%
lat_msh,lon_msh = np.meshgrid(lat_socat,lon_socat,indexing='ij')
path_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/OISSTv2_highres/'
file_sst = path_sst + 'sst.day.mean.'+ str(yr_min) + '.v2.nc'
ds_i = xr.open_dataset(file_sst)
idx_lon_sst = np.where((ds_i.lon.values>=lon_min+360) & (ds_i.lon.values<=lon_max+360))[0]
idx_lat_sst = np.where((ds_i.lat.values>=lat_min) & (ds_i.lat.values<=lat_max))[0]
lon_sst = ds_i.lon[idx_lon_sst].values-360
lat_sst = ds_i.lat[idx_lat_sst].values
sst_i = np.nanmean(ds_i.sst[:,idx_lat_sst,idx_lon_sst].values,axis=0)
mask_gom = np.ones(lat_msh.shape)

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

x1 = -87.0
y1 = 21
x2 = -80.5
y2 =25
a = (y2-y1)/(x2-x1)
b = y2-a*x2

mask_gom[np.isnan(sst_i)] = np.nan
mask_gom[(a*lon_msh+b)>lat_msh] = np.nan
mask_gom[(lon_msh<=-89.75) & (lat_msh>=29.5)] = np.nan
# %% CCMP 
# file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y2003/M01/CCMP_Wind_Analysis_200301_V02.0_L3.5_RSS.nc'
file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y2003/M01/CCMP_Wind_Analysis_200301_V02.0_L3.5_RSS.nc'
# file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v3/y2003/m01/CCMP_Wind_Analysis_200301_V03.0_L4.5.nc'
ds_wind = xr.open_dataset(file_wind)
lon_wind = ds_wind.longitude.values
lat_wind = ds_wind.latitude.values
idxlon = np.where((lon_wind>=lon_min+360) & (lon_wind<=lon_max+360))[0]
idxlat = np.where((lat_wind>=lat_min) & (lat_wind<=lat_max))[0]
lon_wind = lon_wind[idxlon]-360.0
lat_wind = lat_wind[idxlat]
uwnd = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
vwnd = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
wspd = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
for idxyr in range(yr_min, 2019):
    for idxmon in range(1,13):
        # file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        # file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v3/y'+str(idxyr)+'/m'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V03.0_L4.5.nc'
        ds_wind = xr.open_dataset(file_wind)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        uwnd[idx_i,:,:] = ds_wind.uwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        vwnd[idx_i,:,:] = ds_wind.vwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        wspd[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values
if yr_max > 2018:
    idxyr = 2019
    for idxmon in range(1,4):
        # file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        # file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v3/y'+str(idxyr)+'/m'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V03.0_L4.5.nc'
        ds_wind = xr.open_dataset(file_wind)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        uwnd[idx_i,:,:] = ds_wind.uwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        vwnd[idx_i,:,:] = ds_wind.vwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        wspd[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values
    for idxmon in range(5,13):
        # path_wind = '/Volumes/Crucial_4T/data/wind/ccmp/v02.1nrt/Y'+str(2019)+'/M'+str(idxmon).zfill(2)+'/*.nc'
        path_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/v02.1nrt/Y'+str(2019)+'/M'+str(idxmon).zfill(2)+'/*.nc'
        # path_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/v02.1nrt/Y'+str(2019)+'/M'+str(idxmon).zfill(2)+'/*.nc'
        files_wind = np.sort(glob(path_wind))
        uwnd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
        vwnd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
        # wspd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
        for idxf in range(len(files_wind)):
            file_i = files_wind[idxf]
            ds_i = xr.open_dataset(file_i)
            uwnd_i[idxf,:,:,:] = np.copy(ds_i.uwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
            vwnd_i[idxf,:,:,:] = np.copy(ds_i.vwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
            # wspd_i[idxf,:,:,:] = np.copy(ds_i.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
        wspd_i = np.sqrt(uwnd_i**2 + vwnd_i**2)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        uwnd[idx_i,:,:] = np.nanmean(uwnd_i,axis=(0,1))
        vwnd[idx_i,:,:] = np.nanmean(vwnd_i,axis=(0,1))
        wspd[idx_i,:,:] = np.nanmean(wspd_i,axis=(0,1))
    if yr_max > 2019:
        for idxyr in range(2020,yr_max+1):
            for idxmon in range(1,13):
                # path_wind = '/Volumes/Crucial_4T/data/wind/ccmp/v02.1nrt/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/*.nc'
                path_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/v02.1nrt/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/*.nc'
                # path_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/v02.1nrt/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/*.nc'
                files_wind = np.sort(glob(path_wind))
                uwnd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
                vwnd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
                # wspd_i = np.full((len(files_wind),4,len(lat_socat),len(lon_socat)),np.nan)
                for idxf in range(len(files_wind)):
                    file_i = files_wind[idxf]
                    ds_i = xr.open_dataset(file_i)
                    uwnd_i[idxf,:,:,:] = np.copy(ds_i.uwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
                    vwnd_i[idxf,:,:,:] = np.copy(ds_i.vwnd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
                    # wspd_i[idxf,:,:,:] = np.copy(ds_i.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values)
                wspd_i = np.sqrt(uwnd_i**2 + vwnd_i**2)
                idx_i = int((idxyr-yr_min)*12+idxmon-1)
                uwnd[idx_i,:,:] = np.nanmean(uwnd_i,axis=(0,1))
                vwnd[idx_i,:,:] = np.nanmean(vwnd_i,axis=(0,1))
                wspd[idx_i,:,:] = np.nanmean(wspd_i,axis=(0,1))
# %%
# Pressure
file_era5 = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/ERA5/ERA5_single_level_monthly_sst_presure.nc'
ds_era5 = xr.open_dataset(file_era5)
idxt = np.where((ds_era5.time.values>=time_min) & (ds_era5.time.values<=time_max))[0]
p0 = ds_era5.sp[idxt,0,:,:].interp(longitude=lon_socat+360,latitude=lat_socat).values
p0_atm = p0/101325
p0_dbar = p0*0.0001
#%% 
# file_co2air = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/NOAA_MBL_Reference/MBLCO2_24N31N.csv'
# df_co2air = pd.read_csv(file_co2air,header=74,sep='\s+')
# df_xco2air_m = df_co2air.groupby(['year','month']).mean()
# xco2air = df_xco2air_m.value.values[(df_xco2air_m.decimal_date>=yr_min)&(df_xco2air_m.decimal_date<yr_max)]
# xco2air_err = df_xco2air_m.uncertainty.values[(df_xco2air_m.decimal_date>=yr_min)&(df_xco2air_m.decimal_date<yr_max)] #0.07%
# xco2air_3d = np.repeat(np.repeat(np.reshape(xco2air,(len(xco2air),1,1)),len(lat_socat),axis=1),len(lon_socat),axis=2)
# pco2air_mbl = co_xco2topco2(xco2air_3d, sst, sss)
# results = sys(par1=xco2air_3d,par2=2000,par1_type=9,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst,pressure_atmosphere=1,pressure_atmosphere_out=p0_atm,pressure=0,pressure_out=0)
# fco2air_mbl = results['fCO2']
# fco2air = p2fCO2(pco2air,sst, p0)
# pco2air1 = results['pCO2']
# results = sys(par1=pco2air,par2=2000,par1_type=4,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst,pressure_atmosphere=1,pressure_atmosphere_out=p0_atm,pressure=0,pressure_out=0)
# fco2air2 = results['fCO2']
# # a = np.ndarray.flatten(fco2air)
# # b = np.ndarray.flatten(fco2air2)
# # plt.scatter(np.arange(len(a)),a-b)
#%% Carbon Tracker
xco2air = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
for idxyr in range(yr_min,yr_max+1):
    for idxmon in range(1,13):
        file_ct = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CarbonTracker/CT2022GB/CT2022.molefrac_glb3x2_'+str(idxyr)+'-'+str(idxmon).zfill(2)+'.nc'
        ds_ct_i = xr.open_dataset(file_ct)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        xco2air[idx_i,:,:] = ds_ct_i.co2[0,0,:,:].interp(longitude=lon_socat,latitude=lat_socat).values
pco2air = co_xco2topco2(xco2air, sst, sss)
results = sys(par1=xco2air,par2=2000,par1_type=9,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst,pressure_atmosphere=1,pressure_atmosphere_out=1,pressure=0,pressure_out=0)
fco2air = results['fCO2']

pco2air_gv = co_xco2topco2(xco2air_gv, sst, sss)
results = sys(par1=xco2air_gv,par2=2000,par1_type=9,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst,pressure_atmosphere=1,pressure_atmosphere_out=1,pressure=0,pressure_out=0)
fco2air_gv = results['fCO2']
# fco2air[(fco2air>np.percentile(fco2air[~np.isnan(fco2air)],99.9))|(fco2air<np.percentile(fco2air[~np.isnan(fco2air)],0.1))] = np.nan
# fco2air = p2fCO2(pco2air,sst, 0, p0_atm)
#%%
# results = sys(par1=xco2air,par2=2000,par1_type=9,par2_type=1,salinity=sss,temperature=sst,temperature_out=sst,pressure_atmosphere=1,pressure_atmosphere_out=p0_atm,pressure=0,pressure_out=0)
# fco2air2 = results['fCO2']
# a = np.ndarray.flatten(fco2air)
# b = np.ndarray.flatten(fco2air2)
# plt.scatter(range(len(a)),a-b)
#%%
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

data_i = loadmat('./data/area.mat')
A = np.transpose(data_i['A'],(1,0))
area = {}

# k = 1.0
for key in mask.keys():
    area[key] = np.nansum(A*mask[key])
#     plt.pcolor(np.ndarray.flatten(lon_socat),np.ndarray.flatten(lat_socat),np.ndarray.flatten(mask[key]))
#%%
keys = ['txs','las','wfs','nopenw','nopenm1','nopenm2','nopene','cb','ys']
for key in keys:
    x = np.ndarray.flatten(lon_msh*mask[key])
    y = np.ndarray.flatten(lat_msh*mask[key])
    plt.scatter(x,y)
# %%
K0 = co_K0_Weiss(sst,sss)
kt = co_gas_transfer_velocity(sst, wspd, c=0.251, unit="cmhr")
flux = co_co2flux(fco2sw, fco2air, sst, sss, wspd)
flux[(flux>np.percentile(flux[~np.isnan(flux)],99.9))|(flux<np.percentile(flux[~np.isnan(flux)],0.1))] = np.nan

comment_indicator = None
var_sigfig_dict = {}
var_unc_dict = {}
var_unit_dict = {}

for key in area.keys():
    var_sigfig_dict[key] = 3
    var_unc_dict[key] = 0.02
    var_unit_dict[key] = "mol C/m2/yr"

vars_want_to_ana = area.keys()

var_sigfig_dict_ana = {}
var_unc_dict_ana = {}
var_unit_dict_ana = {}
for v in vars_want_to_ana:
    var_sigfig_dict_ana[v] = var_sigfig_dict[v]
    var_unc_dict_ana[v] = var_unc_dict[v]
    var_unit_dict_ana[v] = var_unit_dict[v]
#%%
# #%%
dfco2 = fco2sw - fco2air
df_sst = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_sss = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_wspd = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_fco2sw = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_fco2air = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_dfco2 = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_flux = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
for key in area.keys():
    mask_i = np.repeat(np.reshape(mask[key],(1,len(lat_socat),len(lon_socat))),fco2sw.shape[0],axis=0)
    df_sst[key] = np.nanmean(sst * mask_i,axis=(1,2)).copy()
    df_sss[key] = np.nanmean(sss * mask_i,axis=(1,2)).copy()
    # df_wspd[key] = np.nanmean(wspd * mask_i,axis=(1,2)).copy()
    df_fco2sw[key] = np.nanmean(pco2sw * mask_i,axis=(1,2)).copy()
    df_fco2air[key] = np.nanmean(fco2air * mask_i,axis=(1,2)).copy()
    df_dfco2[key] = np.nanmean(dfco2 * mask_i,axis=(1,2)).copy()
    df_flux[key] = np.nanmean(flux * mask_i,axis=(1,2)).copy()
# %%
# ts_df_dict = datapreprocessing(df_flux[['time','las']],True)
ts_df_dict = datapreprocessing(df_flux,False)
ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict_ana, var_sigfig_dict_ana)

keys = ['txs','las','wfs','nopenw','nopenm1','nopenm2','nopene']
# keys = ['txs','las','wfs','nopenw','nopenm','nopene']
area_ngom = area['ngom']
flux_clim_monthly = pd.DataFrame({'ngom':np.zeros((12,))})
flux_clim_annual = pd.DataFrame({'ngom':[0]})
for key in keys:
    mask_i = np.repeat(np.reshape(mask[key],(1,len(lat_socat),len(lon_socat))),fco2sw.shape[0],axis=0)
    ts_i = np.nanmean(flux * mask_i,axis=(1,2))
    flux_clim_monthly[key] = ts_stats[key]['monthly'][key]['mean'].values
    flux_clim_annual[key] = np.nanmean(flux_clim_monthly[key])

    flux_clim_monthly['ngom'] = flux_clim_monthly['ngom'] + flux_clim_monthly[key]*area[key]/area['ngom']
    flux_clim_annual['ngom'] = flux_clim_annual['ngom'] + flux_clim_annual[key]*area[key]/area['ngom']
if not('nopenm' in keys):
    flux_clim_monthly['nopenm'] = flux_clim_monthly['nopenm1']*area['nopenm1']/area['nopenm'] + flux_clim_monthly['nopenm2']*area['nopenm2']/area['nopenm']
    flux_clim_annual['nopenm'] = flux_clim_annual['nopenm1']*area['nopenm1']/area['nopenm'] + flux_clim_annual['nopenm2']*area['nopenm2']/area['nopenm']

keys = ['cb','ys']
flux_clim_monthly['sgom'] = 0
flux_clim_annual['sgom'] = 0

for key in keys:
    mask_i = np.repeat(np.reshape(mask[key],(1,len(lat_socat),len(lon_socat))),fco2sw.shape[0],axis=0)
    ts_i = np.nanmean(flux * mask_i,axis=(1,2))
    flux_clim_monthly[key] = ts_stats[key]['monthly'][key]['mean'].values
    flux_clim_annual[key] = np.nanmean(flux_clim_monthly[key])

    flux_clim_monthly['sgom'] = flux_clim_monthly['sgom'] + flux_clim_monthly[key]*area[key]/area['sgom']
    flux_clim_annual['sgom'] = flux_clim_annual['sgom'] + flux_clim_annual[key]*area[key]/area['sgom']

flux_clim_monthly['ncoastal'] = flux_clim_monthly['txs']*area['txs']/area['ncoastal'] + flux_clim_monthly['las']*area['las']/area['ncoastal'] + flux_clim_monthly['wfs']*area['wfs']/area['ncoastal']
flux_clim_monthly['nopen'] = flux_clim_monthly['nopenw']*area['nopenw']/area['nopen'] + flux_clim_monthly['nopenm']*area['nopenm']/area['nopen'] + flux_clim_monthly['nopene']*area['nopene']/area['nopen']
flux_clim_monthly['gom'] = flux_clim_monthly['sgom']*area['sgom']/area['gom'] + flux_clim_monthly['ngom']*area['ngom']/area['gom']

flux_clim_annual['ncoastal'] = flux_clim_annual['txs']*area['txs']/area['ncoastal'] + flux_clim_annual['las']*area['las']/area['ncoastal'] + flux_clim_annual['wfs']*area['wfs']/area['ncoastal']
flux_clim_annual['nopen'] = flux_clim_annual['nopenw']*area['nopenw']/area['nopen'] + flux_clim_annual['nopenm']*area['nopenm']/area['nopen'] + flux_clim_annual['nopene']*area['nopene']/area['nopen']
flux_clim_annual['gom'] = flux_clim_annual['sgom']*area['sgom']/area['gom'] + flux_clim_annual['ngom']*area['ngom']/area['gom']

flux_clim_monthly = flux_clim_monthly.reindex(['txs','las','wfs','nopenw','nopenm1','nopenm2','nopenm','nopene','cb','ys','ncoastal','nopen','ngom','sgom','gom'], axis=1)
flux_clim_annual = flux_clim_annual.reindex(['txs','las','wfs','nopenw','nopenm1','nopenm2','nopenm','nopene','cb','ys','ncoastal','nopen','ngom','sgom','gom'], axis=1)
flux_clim_annual.head()

keys = ['txs','las','wfs','nopenw','nopenm1','nopenm2','nopenm','nopene','cb','ys','ngom','sgom']
flux_std_annual = {}
for key in keys:
    flux_std_annual[key] = np.copy(np.nanstd(ts_stats[key]['annual'][key]['mean'].values[:]))

# %%
N_mon = sst.shape[0]
idx_seas = np.full((int(N_mon/4),4),np.nan)
idx_seas[:,0] = np.ndarray.flatten(np.array([np.arange(3,N_mon,12),np.arange(4,N_mon,12),np.arange(5,N_mon,12)]).T-1)
idx_seas[:,1] = np.ndarray.flatten(np.array([np.arange(6,N_mon,12),np.arange(7,N_mon,12),np.arange(8,N_mon,12)]).T-1)
idx_seas[:,2] = np.ndarray.flatten(np.array([np.arange(9,N_mon,12),np.arange(10,N_mon,12),np.arange(11,N_mon,12)]).T-1)
idx_seas[:,3] = np.ndarray.flatten(np.array([np.arange(12,N_mon+1,12),np.arange(1,N_mon,12),np.arange(2,N_mon,12)]).T-1)
idx_seas = np.int32(idx_seas)
dfco2 = fco2sw-fco2air

sst_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
sss_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
wspd_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
dfco2_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
fco2sw_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
uwnd_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)
vwnd_seas_clim = np.full((4,len(lat_socat),len(lon_socat)),np.nan)

for idxsea in range(4):
    sst_seas_clim[idxsea,:,:] = np.nanmean(sst[idx_seas[:,idxsea],:,:],axis=0).copy()
    sss_seas_clim[idxsea,:,:] = np.nanmean(sss[idx_seas[:,idxsea],:,:],axis=0).copy()
    wspd_seas_clim[idxsea,:,:] = np.nanmean(wspd[idx_seas[:,idxsea],:,:],axis=0).copy()
    uwnd_seas_clim[idxsea,:,:] = np.nanmean(uwnd[idx_seas[:,idxsea],:,:],axis=0).copy()
    vwnd_seas_clim[idxsea,:,:] = np.nanmean(vwnd[idx_seas[:,idxsea],:,:],axis=0).copy()
    dfco2_seas_clim[idxsea,:,:] = np.nanmean(dfco2[idx_seas[:,idxsea],:,:],axis=0).copy()
    fco2sw_seas_clim[idxsea,:,:] = np.nanmean(pco2sw[idx_seas[:,idxsea],:,:],axis=0).copy()
#%%
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
#%%
# %%
file_socat = 'data/SOCAT_descrete_GOM.mat'
data_i = loadmat(file_socat)
for key in ['__header__','__version__','__globals__','hdrs']:
    data_i.pop(key)
# colnames = ['Expocode','version','Source_DOI','QC_Flag','yr','mon','day','hh','mm','ss','longitude','latitude','sample_depth','sal','SST','Tequ','PPPP','Pequ','WOA_SSS','NCEP_SLP','ETOPO2_depth','dist_to_land','GVCO2','xCO2water_equ_dry','xCO2water_SST_dry','pCO2water_equ_wet','pCO2water_SST_wet','fCO2water_equ_wet','fCO2water_SST_wet','fCO2rec','fCO2rec_src','fCO2rec_flag']
df_socat_descrete = pd.DataFrame({})
for key in data_i.keys(): 
    df_socat_descrete[key] = np.squeeze(data_i[key])
#QC control
paras_qc = ['fCO2rec','sal','SST']
idxdrop = np.array([])
for para in paras_qc:
    qc_ts = df_socat_descrete['fCO2rec'].values.copy()
    idxdrop = np.append(idxdrop,np.where((qc_ts>np.percentile(qc_ts[~np.isnan(qc_ts)],99.9))|(qc_ts<np.percentile(qc_ts[~np.isnan(qc_ts)],0.1)))[0])
df_socat_qc = df_socat_descrete.drop(index=idxdrop)
df_socat_qc.index = np.arange(df_socat_qc.shape[0])
df_socat_qc['longitude'] = df_socat_qc['longitude']-360
x1 = -83
y1 = 31
x2 = -81
y2 =25.5
a = (y2-y1)/(x2-x1)
b = y2-a*x2
idxdrop = np.where((a*df_socat_qc['longitude'].values+b<df_socat_qc['latitude'].values))[0]
# df_socat_qc.drop(index=idxdrop)

x1 = -87.0
y1 = 21
x2 = -80.5
y2 =25
a = (y2-y1)/(x2-x1)
b = y2-a*x2
idxdrop = np.append(idxdrop,np.where((a*df_socat_qc['longitude'].values+b>df_socat_qc['latitude'].values))[0])
df_socat_qc = df_socat_qc.drop(index=idxdrop)


results = sys(par1=df_socat_qc['fCO2rec'],par2=1000,par1_type=5,par2_type=1,salinity=df_socat_qc['sal'],temperature=df_socat_qc['SST'],temperature_out=df_socat_qc['SST'],pressure=5,pressure_out=5)
df_socat_qc['pco2sw'] = results['pCO2']
# mask_gom[(lon_msh<=-89.75) & (lat_msh>=29.5)] = np.nan
# %% Original
# ts_fco2 = dict({})
# ts_dfco2 = dict({})
# ts_sst = dict({})
# ts_dsst = dict({})
# ts_dfco2_ther = dict({})
# ts_dfco2_nonther = dict({})

# keys = ['txs','las','wfs','nopenw','nopenm','nopene','ys','cb']

# for key in keys:
#     ts_fco2[key] = np.nanmean(fco2sw * mask[key],axis=(1,2))
#     ts_dfco2[key] = np.diff(ts_fco2[key])

#     ts_sst[key] = np.nanmean(sst * mask[key],axis=(1,2))
#     ts_dsst[key] = np.diff(ts_sst[key])
#     ts_dfco2_ther[key] = ts_fco2[key][:-1]*np.exp(0.0423*ts_dsst[key]) - ts_fco2[key][:-1]

#     ts_dfco2_nonther[key] = ts_dfco2[key] - ts_dfco2_ther[key]

# %% - Annual climatology
ts_fco2 = dict({})
ts_dfco2 = dict({})
ts_sst = dict({})
ts_dsst = dict({})
ts_dfco2_ther = dict({})
ts_dfco2_nonther = dict({})

fco2_annualclim = dict({})
sst_annualclim = dict({})

keys = ['txs','las','wfs','nopenw','nopenm','nopene','ys','cb']

for key in keys:
    ts_fco2[key] = np.nanmean(fco2sw * mask[key],axis=(1,2))
    fco2_annualclim[key] = np.nanmean(np.nanmean(np.reshape(ts_fco2[key],[np.int64(len(ts_fco2[key])/12),12]),axis=0))
    ts_dfco2[key] = ts_fco2[key] - fco2_annualclim[key]

    ts_sst[key] = np.nanmean(sst * mask[key],axis=(1,2))
    sst_annualclim[key] = np.nanmean(np.nanmean(np.reshape(ts_sst[key],[np.int64(len(ts_sst[key])/12),12]),axis=0))
    ts_dsst[key] = ts_sst[key] - sst_annualclim[key]

    ts_dfco2_ther[key] = fco2_annualclim[key]*np.exp(0.0423*ts_dsst[key]) - fco2_annualclim[key]

    ts_dfco2_nonther[key] = ts_dfco2[key] - ts_dfco2_ther[key]
# %%
idx_warm = np.array([np.arange(4,fco2sw.shape[0],12),np.arange(5,fco2sw.shape[0],12),np.arange(6,fco2sw.shape[0],12),np.arange(7,fco2sw.shape[0],12),np.arange(8,fco2sw.shape[0],12),np.arange(9,fco2sw.shape[0],12)]).T.flatten() - 1
idx_cold = np.array([np.arange(1,fco2sw.shape[0],12),np.arange(2,fco2sw.shape[0],12),np.arange(3,fco2sw.shape[0],12),np.arange(10,fco2sw.shape[0],12),np.arange(11,fco2sw.shape[0],12),np.arange(12,fco2sw.shape[0]+1,12)]).T.flatten() - 1
idx_cold = idx_cold[:-1]

keys = ['txs','las','wfs']
rnames = ['TXS','LAS','WFS']

plt.figure(figsize=(12,7.5))
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.95,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.3)
for i in range(3):
    key = keys[i]
    x = ts_dfco2[key]
    y_ther = ts_dfco2_ther[key]
    y_nonther = ts_dfco2_nonther[key]

    x_warm = x[idx_warm]
    y_ther_warm = y_ther[idx_warm]
    y_nonther_warm = y_nonther[idx_warm]

    x_cold = x[idx_cold]
    y_ther_cold = y_ther[idx_cold]
    y_nonther_cold = y_nonther[idx_cold]

    idxvalid = (~np.isnan(x_warm)) & (~np.isnan(y_ther_warm))
    lm_ther_warm = sm.OLS(y_ther_warm[idxvalid],sm.add_constant(x_warm[idxvalid])).fit()

    idxvalid = (~np.isnan(x_cold)) & (~np.isnan(y_ther_cold))
    lm_ther_cold = sm.OLS(y_ther_cold[idxvalid],sm.add_constant(x_cold[idxvalid])).fit()

    idxvalid = (~np.isnan(x)) & (~np.isnan(y_ther))
    lm_ther_all = sm.OLS(y_ther[idxvalid],sm.add_constant(x[idxvalid])).fit()

    idxvalid = (~np.isnan(x_warm)) & (~np.isnan(y_nonther_warm))
    lm_nonther_warm = sm.OLS(y_nonther_warm[idxvalid],sm.add_constant(x_warm[idxvalid])).fit()

    idxvalid = (~np.isnan(x_cold)) & (~np.isnan(y_nonther_cold))
    lm_nonther_cold = sm.OLS(y_nonther_cold[idxvalid],sm.add_constant(x_cold[idxvalid])).fit()

    idxvalid = (~np.isnan(x)) & (~np.isnan(y_nonther))
    lm_nonther_all = sm.OLS(y_nonther[idxvalid],sm.add_constant(x[idxvalid])).fit()

    plt.subplot(2,3,i+1)
    plt.scatter(x_warm,y_ther_warm,c='tab:red')
    plt.scatter(x_cold,y_ther_cold,c='tab:blue')
    plt.grid('on',alpha=0.5)
    plt.xlim([-130,130])
    plt.ylim([-130,130])
    plt.plot([-130,130],[-130,130],'k',alpha=0.5)
    plt.xlabel('δ$\it{p}CO_{2total}$ (µatm)')
    plt.ylabel('δ$\it{p}CO_{2therm}$ (µatm)')
    plt.text(-50,-80-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_warm.rsquared) +', p = '+'{:.2f}'.format(lm_ther_warm.pvalues[1]) + ' (summer)',color='tab:red')
    plt.text(-50,-95-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_cold.rsquared) +', p = '+'{:.2f}'.format(lm_ther_cold.pvalues[1]) + ' (winter)',color='tab:blue')
    plt.text(-50,-110-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_all.rsquared) +', p = '+'{:.2f}'.format(lm_ther_all.pvalues[1]) + ' (all)')
    plt.text(-130,135,'('+chr(97+i)+') '+rnames[i] +' thermal vs. total',fontdict={'size':13,'weight':'bold'})

    plt.subplot(2,3,i+4)
    plt.scatter(x_warm,y_nonther_warm,c='tab:red',marker='s')
    plt.scatter(x_cold,y_nonther_cold,c='tab:blue',marker='s')
    plt.grid('on',alpha=0.5)
    plt.xlim([-130,130])
    plt.ylim([-130,130])
    plt.plot([-130,130],[-130,130],'k',alpha=0.5)
    plt.xlabel('δ$\it{p}CO_{2total}$ (µatm)')
    plt.ylabel('δ$\it{p}CO_{2nontherm}$ (µatm)')
    plt.text(-50,-80-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_warm.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_warm.pvalues[1]) + ' (summer)',color='tab:red')
    plt.text(-50,-95-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_cold.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_cold.pvalues[1]) + ' (winter)',color='tab:blue')
    plt.text(-50,-110-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_all.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_all.pvalues[1]) + ' (all)')
    plt.text(-130,135,'('+chr(97+3+i)+') '+rnames[i] +' nonthermal vs. total',fontdict={'size':13,'weight':'bold'})
# plt.tight_layout()
plt.savefig('figs/Fig06A_Coastal_TherNonTher.jpg',dpi=300)
# %%
keys = ['nopenw','nopenm','nopene','ys']
rnames = ['nOPEN-W','nOPEN-M','nOPEN-E','YS']
plt.figure(figsize=(17,7.5))
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.95,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.3)
for i in range(4):
    key = keys[i]
    x = ts_dfco2[key]
    y_ther = ts_dfco2_ther[key]
    y_nonther = ts_dfco2_nonther[key]

    x_warm = x[idx_warm]
    y_ther_warm = y_ther[idx_warm]
    y_nonther_warm = y_nonther[idx_warm]

    x_cold = x[idx_cold]
    y_ther_cold = y_ther[idx_cold]
    y_nonther_cold = y_nonther[idx_cold]

    idxvalid = (~np.isnan(x_warm)) & (~np.isnan(y_ther_warm))
    lm_ther_warm = sm.OLS(y_ther_warm[idxvalid],sm.add_constant(x_warm[idxvalid])).fit()

    idxvalid = (~np.isnan(x_cold)) & (~np.isnan(y_ther_cold))
    lm_ther_cold = sm.OLS(y_ther_cold[idxvalid],sm.add_constant(x_cold[idxvalid])).fit()

    idxvalid = (~np.isnan(x)) & (~np.isnan(y_ther))
    lm_ther_all = sm.OLS(y_ther[idxvalid],sm.add_constant(x[idxvalid])).fit()

    idxvalid = (~np.isnan(x_warm)) & (~np.isnan(y_nonther_warm))
    lm_nonther_warm = sm.OLS(y_nonther_warm[idxvalid],sm.add_constant(x_warm[idxvalid])).fit()

    idxvalid = (~np.isnan(x_cold)) & (~np.isnan(y_nonther_cold))
    lm_nonther_cold = sm.OLS(y_nonther_cold[idxvalid],sm.add_constant(x_cold[idxvalid])).fit()

    idxvalid = (~np.isnan(x)) & (~np.isnan(y_nonther))
    lm_nonther_all = sm.OLS(y_nonther[idxvalid],sm.add_constant(x[idxvalid])).fit()

    plt.subplot(2,4,i+1)
    plt.scatter(x_warm,y_ther_warm,c='tab:red')
    plt.scatter(x_cold,y_ther_cold,c='tab:blue')
    plt.grid('on',alpha=0.5)
    plt.xlim([-130,130])
    plt.ylim([-130,130])
    plt.plot([-130,130],[-130,130],'k',alpha=0.5)
    plt.xlabel('δ$\it{p}CO_{2total}$ (µatm)')
    plt.ylabel('δ$\it{p}CO_{2therm}$ (µatm)')
    plt.text(-50,-80-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_warm.rsquared) +', p = '+'{:.2f}'.format(lm_ther_warm.pvalues[1]) + ' (summer)',color='tab:red')
    plt.text(-50,-95-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_cold.rsquared) +', p = '+'{:.2f}'.format(lm_ther_cold.pvalues[1]) + ' (winter)',color='tab:blue')
    plt.text(-50,-110-10,'R$^2$ = ' + '{:.2f}'.format(lm_ther_all.rsquared) +', p = '+'{:.2f}'.format(lm_ther_all.pvalues[1]) + ' (all)')
    plt.text(-130,135,'('+chr(97+i)+') '+rnames[i] +' thermal vs. total',fontdict={'size':13,'weight':'bold'})

    plt.subplot(2,4,i+5)
    plt.scatter(x_warm,y_nonther_warm,c='tab:red',marker='s')
    plt.scatter(x_cold,y_nonther_cold,c='tab:blue',marker='s')
    plt.grid('on',alpha=0.5)
    plt.xlim([-130,130])
    plt.ylim([-130,130])
    plt.plot([-130,130],[-130,130],'k',alpha=0.5)
    plt.xlabel('δ$\it{p}CO_{2total}$ (µatm)')
    plt.ylabel('δ$\it{p}CO_{2nontherm}$ (µatm)')
    plt.text(-50,-80-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_warm.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_warm.pvalues[1]) + ' (summer)',color='tab:red')
    plt.text(-50,-95-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_cold.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_cold.pvalues[1]) + ' (winter)',color='tab:blue')
    plt.text(-50,-110-10,'R$^2$ = ' + '{:.2f}'.format(lm_nonther_all.rsquared) +', p = '+'{:.2f}'.format(lm_nonther_all.pvalues[1]) + ' (all)')
    plt.text(-130,135,'('+chr(97+4+i)+') '+rnames[i] +' nonthermal vs. total',fontdict={'size':13,'weight':'bold'})
# plt.tight_layout()
plt.savefig('figs/Fig07A_Coastal_TherNonTher.jpg',dpi=300)

# %%
keys = ['txs','las','wfs','nopenw','nopenm','nopene','ys']
rnames = ['TXS','LAS','WFS','nOPEN-W','nOPEN-M','nOPEN-E','YS']

plt.figure(figsize=(14,7))
plt.subplots_adjust(left=0.07,
                    bottom=0.07,
                    right=0.95,
                    top=0.95,
                    wspace=0.3,
                    hspace=0.3)
for i in range(7):
    key = keys[i]
    x = range(1,13)
    y1 = ts_dfco2[key]
    y_ther = ts_dfco2_ther[key]
    y_nonther = ts_dfco2_nonther[key]

    y_mon_clim_total = np.nanmean(np.reshape(y1,[np.int64(len(y1)/12),12]),axis=0)
    y_mon_clim_ther = np.nanmean(np.reshape(y_ther,[np.int64(len(y_ther)/12),12]),axis=0)
    y_mon_clim_nonther = np.nanmean(np.reshape(y_nonther,[np.int64(len(y1)/12),12]),axis=0)
    if i<3:
        plt.subplot(2,4,i+1)
        plt.plot(x,y_mon_clim_total,label='total')
        plt.plot(x,y_mon_clim_ther,label='ther')
        plt.plot(x,y_mon_clim_nonther,label='nonther')
        
        plt.legend()
    else:
        plt.subplot(2,4,i+2)
        plt.plot(x,y_mon_clim_total,label='total')
        plt.plot(x,y_mon_clim_ther,label='ther')
        plt.plot(x,y_mon_clim_nonther,label='nonther')
        plt.legend()
    plt.xlim([0.5,12.5])
    plt.ylim([-100,100])
    plt.text(0.5,105,rnames[i])
plt.savefig('figs/Fig07B_Coastal_TherNonTher.jpg',dpi=300)
# %%
