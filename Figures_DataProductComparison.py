# %%
import xarray as xr
import numpy as np
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
from skimage import io
# 3rd party libraries
import statsmodels.api as sm
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from mpl_toolkits.axes_grid1 import AxesGrid
import cartopy.feature as cfeature
from matplotlib.gridspec import GridSpec
from PyCO2SYS import sys
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
yr_min = 2003
yr_max = 2017
time_min = np.datetime64('2003-01-01')
time_max = np.datetime64('2017-12-31')
# %%
path_obs = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/GOM_Comparison/OBSBased/'
# MPI_SOM-FFN_v2022
file_mpisomfnn = path_obs + 'MPI_SOM-FFN_v2022/MPI_SOM_FFN_2022_NCEI_OCADS.nc'
ds_mpisomfnn = xr.open_dataset(file_mpisomfnn)
idxlon = np.where((ds_mpisomfnn.lon.values>=lon_min) & (ds_mpisomfnn.lon.values<=lon_max))[0]
idxlat = np.where((ds_mpisomfnn.lat.values>=lat_min) & (ds_mpisomfnn.lat.values<=lat_max))[0]
idxtime = np.where((ds_mpisomfnn.time.values>=time_min) & (ds_mpisomfnn.time.values<=time_max))[0]
lon = ds_mpisomfnn.lon.values[idxlon]
lat = ds_mpisomfnn.lat.values[idxlat]
time = ds_mpisomfnn.time.values[idxtime]
pco2_mpisomfnn = ds_mpisomfnn.spco2_smoothed[idxtime,idxlat,idxlon].values
# Jena-MLS
file_jenamls = path_obs + 'Jena-MLS/oc_v2021_pCO2_daily.nc'
ds_jenamls = xr.open_dataset(file_jenamls)
idxlon = np.where((ds_jenamls.lon.values>=lon_min) & (ds_jenamls.lon.values<=lon_max))[0]
idxlat = np.where((ds_jenamls.lat.values>=lat_min) & (ds_jenamls.lat.values<=lat_max))[0]
idxtime = np.where((ds_jenamls.mtime.values>=time_min) & (ds_jenamls.mtime.values<=time_max))[0]
pco2_jenamls = ds_jenamls.pCO2.interp(lon=lon,lat=lat,mtime=time).values
# CMEMS-LSCE-FFNNv2
file_cmems = path_obs + 'CMEMS-LSCE-FFNNv2/2003/dataset-carbon-rep-monthly_20030115T0000Z_P20220930T1545Z.nc'
ds_cmems = xr.open_dataset(file_cmems)
idxlon = np.where((ds_cmems.longitude.values>=lon_min+360) & (ds_cmems.longitude.values<=lon_max+360))[0]
idxlat = np.where((ds_cmems.latitude.values>=lat_min) & (ds_cmems.latitude.values<=lat_max))[0]
pco2_cmems = np.full(pco2_jenamls.shape,np.nan)
for idxyr in range(2003,2018):
    files_i = np.sort(glob(path_obs + 'CMEMS-LSCE-FFNNv2/'+str(idxyr) + '/*.nc'))
    for idxmon in range(0,12):
        file_i = files_i[idxmon]
        ds_i = xr.open_dataset(file_i)
        pco2_cmems[(idxyr-2003)*12+idxmon,:,:] = ds_i.spco2[:,idxlat,idxlon].values
# LDEO-HPD
file_ldeohpd = path_obs + 'LDEO-HPD/LDEO-HPD_spco2_v20210425_1x1_198201-201812.nc'
ds_ldeohpd = xr.open_dataset(file_ldeohpd)
idxlon = np.where((ds_ldeohpd.lon.values>=lon_min+360) & (ds_ldeohpd.lon.values<=lon_max+360))[0]
idxlat = np.where((ds_ldeohpd.lat.values>=lat_min) & (ds_ldeohpd.lat.values<=lat_max))[0]
idxtime = np.where((ds_ldeohpd.time.values>=time_min) & (ds_ldeohpd.time.values<=time_max))[0]
pco2_ldeohpd = ds_ldeohpd.spco2_filled[:,idxtime,idxlat,idxlon].values
pco2_ldeohpd = np.nanmean(pco2_ldeohpd,axis=0)
# NIES-NN
file_niesnn = path_obs + 'NIES-NN/nies.nn.sfco2.1980-2020.ver.2022.2.nc'
ds_niesnn = xr.open_dataset(file_niesnn)
idxlon = np.where((ds_niesnn.lon.values>=lon_min+360) & (ds_niesnn.lon.values<=lon_max+360))[0]
idxlat = np.where((ds_niesnn.lat.values>=lat_min) & (ds_niesnn.lat.values<=lat_max))[0]
idxtime = np.where((ds_niesnn.time.values>=time_min) & (ds_niesnn.time.values<=time_max))[0]
pco2_niesnn = ds_niesnn.sfco2[idxtime,idxlat,idxlon].values
# JMA-MLR
file_jmamlr = path_obs + 'JMA-MLR/JMA_co2map_'+str(2003)+'.nc'
ds_jmamlr= xr.open_dataset(file_jmamlr,decode_times=False)
idxlon = np.where((ds_jmamlr.lon.values>=lon_min+360) & (ds_jmamlr.lon.values<=lon_max+360))[0]
idxlat = np.where((ds_jmamlr.lat.values>=lat_min) & (ds_jmamlr.lat.values<=lat_max))[0]
pco2_jmamlr = np.full(pco2_niesnn.shape,np.nan)
for idxyr in range(2003,2018):
    file_jmamlr = path_obs + 'JMA-MLR/JMA_co2map_'+str(idxyr)+'.nc'
    ds_i = xr.open_dataset(file_jmamlr,decode_times=False)
    pco2_jmamlr[(idxyr-2003)*12:(idxyr-2003)*12+12,:,:] = np.copy(ds_i.pCO2s[:,idxlat,idxlon].values)

# OS-ETHZ-GRaCER
file_osethzgracer = path_obs + 'OS-ETHZ-GRaCER/OceanSODA-ETHZ_GRaCER_v2021a_1982-2020.nc'
ds_osethzgracer = xr.open_dataset(file_osethzgracer)
idxlon = np.where((ds_osethzgracer.lon.values>=lon_min) & (ds_osethzgracer.lon.values<=lon_max))[0]
idxlat = np.where((ds_osethzgracer.lat.values>=lat_min) & (ds_osethzgracer.lat.values<=lat_max))[0]
idxtime = np.where((ds_osethzgracer.time.values>=time_min) & (ds_osethzgracer.time.values<=time_max))[0]
pco2_osethzgracer = ds_osethzgracer.spco2[idxtime,idxlat,idxlon].values
# CHEN19
lon_chen19 = np.linspace(-98,-79,2090)
lat_chen19 = np.linspace(18,31,1430)
path_chen19 = path_obs + 'CHEN19/monthly_hu/'
files_chen19 = np.sort(glob(path_chen19 + '*.tif'))
pco2_chen19 = np.full((len(files_chen19),1430,2090),np.nan)
for idxf in range(len(files_chen19)):
    file_i = files_chen19[idxf]
    pco2_chen19[idxf] = np.copy(io.imread(file_i))
pco2_chen19 = np.flip(pco2_chen19,axis=1)
pco2_chen19 = pco2_chen19[6:]
pco2_chen19[pco2_chen19<1] = np.nan
step = 4
pco2_chen19 = pco2_chen19[:,::step,::step]
lat_chen19 = lat_chen19[::step]
lon_chen19 = lon_chen19[::step]
pco2_chen19_clim = np.nanmean(pco2_chen19,axis=0)
# 
# %%
pco2_ensemble = np.full(np.append(np.array([7,]),np.array(pco2_osethzgracer.shape)),np.nan)
pco2_ensemble[0] = pco2_mpisomfnn
pco2_ensemble[1] = pco2_jenamls
pco2_ensemble[2] = pco2_cmems
pco2_ensemble[3] = pco2_ldeohpd
pco2_ensemble[4] = pco2_niesnn
pco2_ensemble[5] = pco2_jmamlr
pco2_ensemble[6] = pco2_osethzgracer
# %%
pco2_ensemble_mean = np.nanmean(pco2_ensemble,axis=0)
pco2_ensemble_clim = np.nanmean(pco2_ensemble,axis=1)
pco2_ensemble_avg = np.nanmean(pco2_ensemble_clim,axis=0)
pco2_ensemble_std = np.nanstd(pco2_ensemble_clim,axis=0)
#%% Read SOCAT Coastal
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
mask_esemble = get_gom_mask(lon,lat)
mask_chen19 = get_gom_mask(lon_chen19,lat_chen19)
mask_socat = get_gom_mask(lon_socat,lat_socat)
data_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/OISSTv2_highres/sst.day.mean.1981.v2.nc'
ds_i = xr.open_dataset(data_sst)
sst_i = ds_i.sst.interp(time=ds_i.time.values[0],lon=lon+360,lat=lat).values
for key in mask_esemble.keys():
    mask_esemble[key][np.isnan(sst_i)] = np.nan
#%%
time_all = []
for idxyr in range(2003,2018):
    for idxmon in range(1,13):
        time_all = np.append(time_all,str(idxyr)+'-'+str(idxmon).zfill(2)+'-15')

var_sigfig_dict = {}
var_unc_dict = {}
var_unit_dict = {}

var_sigfig_dict['pco2'] = 1
var_unc_dict['pco2'] = 5
var_unit_dict['pco2'] = "µatm"
#%%
# tr_chen19 = np.full((len(lat_chen19),len(lon_chen19)),np.nan)
# clim_chen19 = np.full((len(lat_chen19),len(lon_chen19)),np.nan)
# p_chen19 = np.full((len(lat_chen19),len(lon_chen19)),np.nan)
# for idxlat in range(len(lat_chen19)):
#     for idxlon in range(len(lon_chen19)):
#         ts = pco2_chen19[:,idxlat,idxlon]
#         if (np.sum(~np.isnan(ts))>100) & (mask_chen19['gom'][idxlat,idxlon] == 1):
#             print([idxlat,idxlon])
#             df_i = pd.DataFrame({'time':time_all,
#                                 'pco2':ts})
#             ts_df_dict = datapreprocessing(df_i,True)
#             ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict)
#             clim_chen19[idxlat,idxlon] = np.copy(np.nanmean(ts_stats['pco2']['monthly']['pco2']['mean'].values))
#             tr_chen19[idxlat,idxlon] = np.copy(np.float32(var_trends['pco2']['slope_str']))
#             p_chen19[idxlat,idxlon] = wls_model_dict['pco2']['model'].pvalues[1]

# ds_chen = xr.Dataset(
#     data_vars=dict(
#         tr_chen19=(["lat", "lon"], tr_chen19),
#         clim_chen19=(["lat", "lon"], clim_chen19),
#         p_chen19=(["lat", "lon"], p_chen19),
#     ),
#     coords=dict(
#         lon=lon_chen19,
#         lat=lat_chen19,
#     ),
# )
# ds_chen.to_netcdf('./data/chen19.nc')
#%%
ds_chen = xr.open_dataset('./data/chen19.nc')
lon_chen19 = ds_chen.lon.values
lat_chen19 = ds_chen.lat.values
clim_chen19 = ds_chen.clim_chen19.values
tr_chen19 = ds_chen.tr_chen19.values
p_chen19 = ds_chen.p_chen19.values
#%%
bin = 2
tr_socat = np.full((len(lat_socat),len(lon_socat)),np.nan)
clim_socat = np.full((len(lat_socat),len(lon_socat)),np.nan)
p_socat = np.full((len(lat_socat),len(lon_socat)),np.nan)
for idxlat in range(len(lat_socat)):
    for idxlon in range(len(lon_socat)):
        ts = np.nanmean(pco2sw[:,idxlat-bin:idxlat+bin+1,idxlon-bin:idxlon+bin+1],axis=(1,2))
        time_i = time_all[~np.isnan(ts)]
        if (np.sum(~np.isnan(ts))>=20) & (~np.isnan(mask_socat['gom'][idxlat,idxlon])):
            if (pd.to_datetime(time_i).year[-1] - pd.to_datetime(time_i).year[0]>10):
                # print([idxlat,idxlon])
                df_i = pd.DataFrame({'time':time_all,
                                    'pco2':ts})
                ts_df_dict = datapreprocessing(df_i,True)
                ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict)
                clim_socat[idxlat,idxlon] = np.copy(np.nanmean(ts_stats['pco2']['monthly']['pco2']['mean'].values))
                tr_socat[idxlat,idxlon] = np.copy(np.float32(var_trends['pco2']['slope_str']))
                p_socat[idxlat,idxlon] = wls_model_dict['pco2']['model'].pvalues[1]
# %%
tr_ensemble = np.full((len(lat),len(lon)),np.nan)
clim_ensemble = np.full((len(lat),len(lon)),np.nan)
p_ensemble = np.full((len(lat),len(lon)),np.nan)

for idxlat in range(len(lat)):
    for idxlon in range(len(lon)):
        ts = pco2_ensemble_mean[:,idxlat,idxlon]
        if (np.sum(~np.isnan(ts))>100) & (~np.isnan(mask_esemble['gom'][idxlat,idxlon])):
            df_i = pd.DataFrame({'time':time_all,
                                'pco2':ts})
            ts_df_dict = datapreprocessing(df_i,True)
            ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict)
            clim_ensemble[idxlat,idxlon] = np.copy(np.nanmean(ts_stats['pco2']['monthly']['pco2']['mean'].values))
            tr_ensemble[idxlat,idxlon] = np.copy(np.float32(var_trends['pco2']['slope_str']))
            p_ensemble[idxlat,idxlon] = wls_model_dict['pco2']['model'].pvalues[1]

# %%
tr_ensemble_sep = np.full((7,len(lat),len(lon)),np.nan)
clim_ensemble_sep = np.full((7,len(lat),len(lon)),np.nan)
p_ensemble_sep = np.full((7,len(lat),len(lon)),np.nan)

for idx_prod in range(7):
    for idxlat in range(len(lat)):
        for idxlon in range(len(lon)):
            ts = pco2_ensemble[idx_prod,:,idxlat,idxlon]
            if (np.sum(~np.isnan(ts))>100) & (~np.isnan(mask_esemble['gom'][idxlat,idxlon])):
                df_i = pd.DataFrame({'time':time_all,
                                    'pco2':ts})
                ts_df_dict = datapreprocessing(df_i,True)
                ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict)
                clim_ensemble_sep[idx_prod,idxlat,idxlon] = np.copy(np.nanmean(ts_stats['pco2']['monthly']['pco2']['mean'].values))
                tr_ensemble_sep[idx_prod,idxlat,idxlon] = np.copy(np.float32(var_trends['pco2']['slope_str']))
                p_ensemble_sep[idx_prod,idxlat,idxlon] = wls_model_dict['pco2']['model'].pvalues[1]
# %%
tr_ensemble_all = np.full((7,len(lat),len(lon)),np.nan)
for idxproduct in range(7):
    for idxlat in range(len(lat)):
        for idxlon in range(len(lon)):
            ts = pco2_ensemble[idxproduct,:,idxlat,idxlon]
            if (np.sum(~np.isnan(ts))>100) & (~np.isnan(mask_esemble['gom'][idxlat,idxlon]) ):
                df_i = pd.DataFrame({'time':time_all,
                                    'pco2':ts})
                ts_df_dict = datapreprocessing(df_i,True)
                ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict, var_sigfig_dict)
                tr_ensemble_all[idxproduct,idxlat,idxlon] = np.copy(np.float32(var_trends['pco2']['slope_str']))
tr_ensemble_std = np.nanstd(tr_ensemble_all,axis=0)
# %% Figure 08
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
fig = plt.figure(figsize=(11.5,7))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(2,2),
                axes_pad=(1.2,0.7),
                cbar_location='right',
                cbar_mode='each',
                cbar_pad=0.3,
                cbar_size='2.5%',
                label_mode='')  # note the empty label_mode
for i, ax in enumerate(axgr):
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
    if i == 0:
        p1 = ax.pcolor(lon_socat, lat_socat, clim_socat*mask_socat['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',14), clim=[320,460],shading='interp')
        titlefig = '(a) SOCAT'
    elif i == 1:
        p1 = ax.pcolor(lon_chen19, lat_chen19, clim_chen19*mask_chen19['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',14), clim=[320,460],shading='interp')
        titlefig = '(b) CHEN19'
    elif i==2:
        p2 = ax.pcolor(lon, lat, clim_ensemble*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',14), clim=[320,460],shading=None)
        titlefig = '(c) Ensemble Avg. (7 products)'
    else:
        p3 = ax.pcolor(lon, lat, pco2_ensemble_std*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('OrRd',10), clim=[0,20],shading='interp')
        titlefig = '(d) Ensemble Std. (7 products)'
    ax.text(-99,31.5,titlefig,fontdict={'size':13,'weight':'bold'})

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # x1 = -87.0
    # y1 = 21
    # x2 = -80.5
    # y2 =25
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')
cb = axgr.cbar_axes[0].colorbar(p1,extend='both',label='pCO$_2$ (µatm)')
cb = axgr.cbar_axes[1].colorbar(p1,extend='both',label='pCO$_2$ (µatm)')
cb = axgr.cbar_axes[2].colorbar(p2,extend='both',label='pCO$_2$ (µatm)')
cb = axgr.cbar_axes[3].colorbar(p3,extend='max',label='pCO$_2$ (µatm)',ticks=np.arange(0,21,2))
plt.tight_layout()
plt.savefig('figs/Fig08_pCO2swAvg_ComparisonMap.jpg',dpi=300)
plt.show()
# %% Figure 09
tr_air = 0
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
fig = plt.figure(figsize=(11.5,7))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(2,2),
                axes_pad=(1.2,0.7),
                cbar_location='right',
                cbar_mode='each',
                cbar_pad=0.3,
                cbar_size='2.5%',
                label_mode='')  # note the empty label_mode
for i, ax in enumerate(axgr):
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
    if i == 0:
        p0 = ax.pcolor(lon_socat, lat_socat, tr_socat*mask_socat['gom']-tr_air, transform=projection, cmap=plt.get_cmap('RdBu_r',12), clim=[-3,3],shading='interp')
        y,x = np.meshgrid(lat_socat,lon_socat,indexing='ij')
        x = x*mask_socat['gom']
        y = y*mask_socat['gom']
        y[p_socat>0.05] = np.nan
        x[p_socat>0.05] = np.nan
        y[np.isnan(p_socat)] = np.nan
        x[np.isnan(p_socat)] = np.nan
        dens = 1
        ax.scatter(x[::dens],y[::dens],5,c='k',alpha=0.2)
        # p0 = ax.contourf(lon_socat, lat_socat, tr_socat*mask_socat['gom']-tr_air, transform=projection, cmap='RdYlBu_r', clim=[-3,3],shading='interp',levels=np.linspace(-3,3,13))
        titlefig = '(a) SOCAT'
    elif i == 1:
        p1 = ax.pcolor(lon_chen19, lat_chen19, tr_chen19*mask_chen19['gom']-tr_air, transform=projection, cmap=plt.get_cmap('RdBu_r',12), clim=[-3,3],shading='interp')
        # p1 = ax.contourf(lon_chen19, lat_chen19, tr_chen19*mask_chen19['gom']-tr_air, transform=projection, cmap='RdBu_r', clim=[-3,3],shading='interp',levels=np.linspace(-3,3,13),extend='both')
        y,x = np.meshgrid(lat_chen19,lon_chen19,indexing='ij')
        x = x*mask_chen19['gom']
        y = y*mask_chen19['gom']
        y[p_chen19>0.05] = np.nan
        x[p_chen19>0.05] = np.nan
        dens = 6
        ax.scatter(x[::dens],y[::dens],1,c='k',alpha=0.1)
        # p1 = ax.contourf(lon_chen19, lat_chen19, tr_chen19*mask_chen19['gom']-tr_air, transform=projection, cmap='RdBu_r', clim=[-3,3],shading='interp',levels=np.linspace(-3,3,13))
        titlefig = '(b) CHEN19'
    elif i==2:
        p2 = ax.pcolor(lon, lat, tr_ensemble*mask_esemble['gom']-tr_air, transform=projection, cmap=plt.get_cmap('RdBu_r',12), clim=[-3,3],shading=None)
        # p2 = ax.contourf(lon, lat, tr_ensemble*mask_esemble['gom']-tr_air, transform=projection, cmap='RdBu_r', clim=[-3,3],shading='flat',levels=np.linspace(-3,3,13),extend='both')
        y,x = np.meshgrid(lat,lon,indexing='ij')
        x = x*mask_esemble['gom']
        y = y*mask_esemble['gom']
        y[p_ensemble>0.05] = np.nan
        x[p_ensemble>0.05] = np.nan
        dens = 1
        ax.scatter(x[::dens],y[::dens],5,c='k',alpha=0.5)
        # p2 = ax.contourf(lon, lat, tr_ensemble*mask['gom']-tr_air, transform=projection, cmap='RdBu_r', clim=[-3,3],shading=None,levels=np.linspace(-3,3,13))
        titlefig = '(c) Ensemble Avg. (7 products)'
    else:
        p3 = ax.pcolor(lon, lat, tr_ensemble_std*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('OrRd',9), clim=[0,1],shading='interp')
        # p3 = ax.contourf(lon, lat, tr_ensemble_std*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',10), clim=[-1,1],shading='flat',levels=np.linspace(-1,1,11),extend='both')
        titlefig = '(d) Ensemble Std. (7 products)'
    ax.text(-99,31.5,titlefig,fontdict={'size':13,'weight':'bold'})

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # x1 = -87.0
    # y1 = 21
    # x2 = -80.5
    # y2 =25
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')
cb = axgr.cbar_axes[0].colorbar(p1,extend='both',label='pCO$_2$ trend (µatm/yr)',ticks=np.arange(-3,3.1,1))
cb = axgr.cbar_axes[1].colorbar(p1,extend='both',label='pCO$_2$ trend (µatm/yr)',ticks=np.arange(-3,3.1,1))
cb = axgr.cbar_axes[2].colorbar(p2,extend='both',label='pCO$_2$ trend (µatm/yr)',ticks=np.arange(-3,3.1,1))
cb = axgr.cbar_axes[3].colorbar(p3,extend='max',label='std. of pCO$_2$ trend (µatm/yr)',ticks=np.linspace(0,1,11))
plt.tight_layout()
plt.savefig('figs/Fig09_pCO2swTrend_ComparisonMap.jpg',dpi=300)
plt.show()
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
fco2sw_socat = ds_socat.fCO2rec[idx_time_socat,idx_lat_socat,idx_lon_socat].values

fco2sw[(fco2sw<1) | (sst<-4)| (sss<0)] = np.nan
fco2sw[(fco2sw>np.percentile(fco2sw[~np.isnan(fco2sw)],99.9))|(fco2sw<np.percentile(fco2sw[~np.isnan(fco2sw)],0.1))] = np.nan
# %% Read OISST
file_sst = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/sst/sst.mnmean.nc'
ds_sst = xr.open_dataset(file_sst)
idx_time_sst = np.where((ds_sst.time.values>=time_min) & (ds_sst.time.values<=time_max))[0]
sst_chen = ds_sst.sst.interp(lon=lon_chen19+360,lat=lat_chen19).values[idx_time_sst]
sst_socat = ds_sst.sst.interp(lon=lon_socat+360,lat=lat_socat).values[idx_time_sst]
sst_ensemble = ds_sst.sst.interp(lon=lon+360,lat=lat).values[idx_time_sst]
#%%
sss_chen = np.full((12,len(lat_chen19),len(lon_chen19)),np.nan)
sss_socat = np.full((12,len(lat_socat),len(lon_socat)),np.nan)
sss_ensemble = np.full((12,len(lat),len(lon)),np.nan)
for idxmon in range(12):
    file_i = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/WOA18/2005-2017-monthly_salinity/woa18_A5B7_s'+str(idxmon).zfill(2)+'_04.nc'
    ds_i = xr.open_dataset(file_i,decode_times=False)
    sss_chen[idxmon] = ds_i.s_an.interp(lon=lon_chen19,lat=lat_chen19,depth=0).values
    sss_socat[idxmon] = ds_i.s_an.interp(lon=lon_socat,lat=lat_socat,depth=0).values
    sss_ensemble[idxmon] = ds_i.s_an.interp(lon=lon,lat=lat,depth=0).values
sss_chen = np.repeat(sss_chen,int(sst_chen.shape[0]/12),axis=0)
sss_socat = np.repeat(sss_socat,int(sst_chen.shape[0]/12),axis=0)
sss_ensemble = np.repeat(sss_ensemble,int(sst_chen.shape[0]/12),axis=0)
#%%
results = sys(par1=fco2sw,par2=1000,par1_type=5,par2_type=1,salinity=sss_socat,temperature=sst_socat,temperature_out=sst_socat)
pco2_socat = results['pCO2']
pco2_socat[pco2sw>2000] = np.nan
# %%
xco2air_socat = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
xco2air_chen = np.full(((yr_max-yr_min+1)*12,len(lat_chen19),len(lon_chen19)),np.nan)
xco2air_ensemble = np.full(((yr_max-yr_min+1)*12,len(lat),len(lon)),np.nan)
for idxyr in range(yr_min,yr_max+1):
    for idxmon in range(1,13):
        file_ct = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/carbon/CarbonTracker/CT2022GB/CT2022.molefrac_glb3x2_'+str(idxyr)+'-'+str(idxmon).zfill(2)+'.nc'
        ds_ct_i = xr.open_dataset(file_ct)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        xco2air_socat[idx_i,:,:] = ds_ct_i.co2[0,0,:,:].interp(longitude=lon_socat,latitude=lat_socat).values
        xco2air_chen[idx_i,:,:] = ds_ct_i.co2[0,0,:,:].interp(longitude=lon_chen19,latitude=lat_chen19).values
        xco2air_ensemble[idx_i,:,:] = ds_ct_i.co2[0,0,:,:].interp(longitude=lon,latitude=lat).values
pco2air_chen = co_xco2topco2(xco2air_chen, sst_chen, sss_chen)
pco2air_socat = co_xco2topco2(xco2air_socat, sst_socat, sss_socat)
pco2air_ensemble = co_xco2topco2(xco2air_ensemble, sst_ensemble, sss_ensemble)
#%%
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
wspd_socat = np.full(((yr_max-yr_min+1)*12,len(lat_socat),len(lon_socat)),np.nan)
wspd_chen = np.full(((yr_max-yr_min+1)*12,len(lat_chen19),len(lon_chen19)),np.nan)
wspd_ensemble = np.full(((yr_max-yr_min+1)*12,len(lat),len(lon)),np.nan)
for idxyr in range(yr_min, 2018):
    for idxmon in range(1,13):
        # file_wind = '/Volumes/Crucial_4T/data/wind/ccmp/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v2/Y'+str(idxyr)+'/M'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V02.0_L3.5_RSS.nc'
        # file_wind = '/Users/zelun/Library/CloudStorage/OneDrive-Personal/data/winds/CCMP/monthly_v3/y'+str(idxyr)+'/m'+str(idxmon).zfill(2)+'/CCMP_Wind_Analysis_'+str(idxyr)+str(idxmon).zfill(2)+'_V03.0_L4.5.nc'
        ds_wind = xr.open_dataset(file_wind)
        idx_i = int((idxyr-yr_min)*12+idxmon-1)
        wspd_socat[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon_socat+360,latitude=lat_socat).values
        wspd_chen[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon_chen19+360,latitude=lat_chen19).values
        wspd_ensemble[idx_i,:,:] = ds_wind.wspd.interp(longitude=lon+360,latitude=lat).values
# %%
flux_chen = co_co2flux(pco2_chen19,pco2air_chen,sst_chen,sss_chen,wspd_chen)
flux_socat = co_co2flux(pco2_socat,pco2air_socat,sst_socat,sss_socat,wspd_socat)
flux_ensemble = co_co2flux(np.nanmean(pco2_ensemble,axis=0),pco2air_ensemble,sst_ensemble,sss_ensemble,wspd_ensemble)
#%%
data_i = loadmat('./data/area.mat')
A = np.transpose(data_i['A'],(1,0))
area = {}

# k = 1.0
for key in mask_socat.keys():
    area[key] = np.nansum(A*mask_socat[key])

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
df_flux = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
for key in area.keys():
    df_flux[key] = np.nanmean(flux_socat * mask_socat[key],axis=(1,2)).copy()

ts_df_dict = datapreprocessing(df_flux,False)
ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict, var_unc_dict_ana, var_sigfig_dict_ana)

keys = ['txs','las','wfs','nopenw','nopenm1','nopenm2','nopene']
# keys = ['txs','las','wfs','nopenw','nopenm','nopene']
area_ngom = area['ngom']
flux_clim_monthly = pd.DataFrame({'ngom':np.zeros((12,))})
flux_clim_annual = pd.DataFrame({'ngom':[0]})
for key in keys:
    mask_i = np.repeat(np.reshape(mask_socat[key],(1,len(lat_socat),len(lon_socat))),fco2sw.shape[0],axis=0)
    ts_i = np.nanmean(flux_socat * mask_i,axis=(1,2))
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
    mask_i = np.repeat(np.reshape(mask_socat[key],(1,len(lat_socat),len(lon_socat))),fco2sw.shape[0],axis=0)
    ts_i = np.nanmean(flux_socat * mask_i,axis=(1,2))
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
# flux SOCAT, nGOM = 0.29 + 0.54
df_flux_chen = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_flux_chen['ngom'] = np.nanmean(flux_chen * mask_chen19['ngom'],axis=(1,2)).copy()
df_flux_chen['gom'] = np.nanmean(flux_chen * mask_chen19['gom'],axis=(1,2)).copy()

ts_df_dict_chen = datapreprocessing(df_flux_chen,False)
ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict_chen, var_unc_dict_ana, var_sigfig_dict_ana)

flux_avg_chen_ngom = np.nanmean(ts_stats['ngom']['monthly']['ngom']['mean'].values)
flux_std_chen_ngom = np.nanstd(ts_stats['ngom']['annual']['ngom']['mean'].values)
# -0.28 + 0.23
flux_avg_chen_gom = np.nanmean(ts_stats['gom']['monthly']['gom']['mean'].values)
flux_std_chen_gom = np.nanstd(ts_stats['gom']['annual']['gom']['mean'].values)
# -0.16 + 0.23
# %%
df_flux_ensemble = pd.DataFrame({'time':time.strftime('%Y-%m-%d')})
df_flux_ensemble['ngom'] = np.nanmean(flux_ensemble * mask_esemble['ngom'],axis=(1,2)).copy()
df_flux_ensemble['gom'] = np.nanmean(flux_ensemble * mask_esemble['gom'],axis=(1,2)).copy()

ts_df_dict_ensemble = datapreprocessing(df_flux_ensemble,False)
ts_stats, var_trends, wls_model_dict, TDTi_dict = calc_all_stats(ts_df_dict_ensemble, var_unc_dict_ana, var_sigfig_dict_ana)

flux_avg_ensemble_ngom = np.nanmean(ts_stats['ngom']['monthly']['ngom']['mean'].values)
flux_std_ensemble_ngom = np.nanstd(ts_stats['ngom']['annual']['ngom']['mean'].values)
# -0.15 + 0.05
flux_avg_ensemble_gom = np.nanmean(ts_stats['gom']['monthly']['gom']['mean'].values)
flux_std_ensemble_gom = np.nanstd(ts_stats['gom']['annual']['gom']['mean'].values)
# -0.00 + 0.03
# %%
# nGOM
a = np.mean([flux_clim_annual['ngom'].values[0],flux_avg_chen_ngom,flux_avg_ensemble_ngom])
b = 1/3 * np.sqrt(flux_std_annual['ngom']**2 + flux_std_chen_ngom**2 + flux_std_ensemble_ngom**2)
print('Mean flux of nGOM is '+ '{:.2f}'.format(a) + '±' '{:.2f}'.format(b))
# GOM
a = np.mean([flux_avg_chen_gom,flux_avg_ensemble_gom])
b = 1/2 * np.sqrt(flux_std_chen_gom**2 + flux_std_ensemble_gom**2)
print('Mean flux of GOM is '+ '{:.2f}'.format(a) + '±' '{:.2f}'.format(b))
# %% Figure A08
names_prod = ['MPI_SOM-FFN','Jena-MLS','CMEMS-LSCE-FFNN','LDEO-HPD','NIES-NN','JMA-MLR','OS-ETHZ-GRaCER']
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
fig = plt.figure(figsize=(11.5,9))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(3,3),
                axes_pad=(0.1,0.5),
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.2,
                cbar_size='1.5%',
                label_mode='')  # note the empty label_mode
for i, ax in enumerate(axgr):
    if i < 7:
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
        
        p0 = ax.pcolor(lon, lat, clim_ensemble_sep[i]*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',14), clim=[320,460],shading=None)# titlefig = '(a) SOCAT'
        titlefig = '(' + chr(97+i) + ') ' + names_prod[i]
        ax.text(-99,31.5,titlefig,fontdict={'size':13,'weight':'bold'})

        # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
        # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
        # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
        # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
        # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
        # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')

        # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
        # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
        # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
        # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
        # x1 = -87.0
        # y1 = 21
        # x2 = -80.5
        # y2 =25
        # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
        # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')
    else:
        ax.set_axis_off()
cb = axgr.cbar_axes[0].colorbar(p0,extend='both',label='pCO$_2$ (µatm)')

plt.tight_layout()
plt.savefig('figs/FigA08_pCO2swAvg_ComparisonMap.jpg',dpi=300)
plt.show()

# %% FigA09
tr_air = 0
projection = ccrs.PlateCarree()
axes_class = (GeoAxes, dict(map_projection=projection))
fig = plt.figure(figsize=(11.5,9))
axgr = AxesGrid(fig, 111, axes_class=axes_class,
                nrows_ncols=(3,3),
                axes_pad=(0.1,0.5),
                cbar_location='right',
                cbar_mode='single',
                cbar_pad=0.2,
                cbar_size='1.5%',
                label_mode='')  # note the empty label_mode
for i, ax in enumerate(axgr):
    if i < 7:
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
        
        p0 = ax.pcolor(lon, lat, tr_ensemble_sep[i]*mask_esemble['gom'], transform=projection, cmap=plt.get_cmap('RdYlBu_r',24), clim=[-3,3],shading=None)# titlefig = '(a) SOCAT'
        titlefig = '(' + chr(97+i) + ') ' + names_prod[i]
        ax.text(-99,31.5,titlefig,fontdict={'size':13,'weight':'bold'})
        
        # ax.text(-99,31.5,titlefig,fontdict={'size':13,'weight':'bold'})

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')

    # ax.plot([-97.8,-82,],[b_nsgom,b_nsgom],'k',linewidth=1)
    # ax.plot([b_txs_las,b_txs_las,],[b_nsgom,29.8],'k',linewidth=1)
    # ax.plot([b_las_wfs,b_las_wfs,],[b_nsgom,30.1],'k',linewidth=1)
    # ax.plot([b_cb_ys,b_cb_ys,],[21.2,b_nsgom],'k',linewidth=1)
    # x1 = -87.0
    # y1 = 21
    # x2 = -80.5
    # y2 =25
    # ax.plot([x1,x2],[y1,y2],'k',linewidth=1)
    # ax.contour(lon,lat,depth*mask['ngom'],levels=[-200],colors='k',linewidth=1,linestyles='-')
    else:
        ax.set_axis_off()
cb = axgr.cbar_axes[0].colorbar(p0,extend='both',label='pCO$_2$ trend (µatm/yr)',ticks=np.arange(-3,3.1,0.25))
plt.tight_layout()
plt.savefig('figs/FigA09_pCO2swTrend_ComparisonMap.jpg',dpi=300)
plt.show()
# %%
