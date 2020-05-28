import datetime
import os

import numpy as np
import pandas as pd

import urllib, json


def _jhu_to_iso(fp_csv:str) -> pd.DataFrame:
    """Convert Johns Hopkins University dataset to nicely formatted DataFrame.

    Drops Lat/Long columns and reformats to a multi-index of (country, state).

    Parameters
    ----------
    fp_csv : string

    Returns
    -------
    : pandas.DataFrame
    """
    df = pd.read_csv(fp_csv, sep=',')
    # change columns & index
    df = df.drop(columns=['Lat', 'Long']).rename(columns={
        'Province/State': 'state',
        'Country/Region': 'country'
    })
    df = df.set_index(['country', 'state'])
    # datetime columns
    df.columns = [datetime.datetime.strptime(d, '%m/%d/%y') for d in df.columns]
    return df


def get_jhu_cdr(
        country:str, state:str,
        fp_confirmed:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv',
        fp_deaths:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv',
        fp_recovered:str='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv',
    ) -> pd.DataFrame:
    """Gets confirmed, deaths and recovered Johns Hopkins University dataset as a DataFrame with datetime index.

    Parameters
    ----------
    country : string
        name of the country (the "Country/Region" column), can be None if state is set
    state : string
        name of the state (the "Province/State" column), can be None if country is set
    fp_confirmed : string
        filepath or URL pointing to the original CSV of global confirmed cases
    fp_deaths : string
        filepath or URL pointing to the original CSV of global deaths
    fp_recovered : string
        filepath or URL pointing to the original CSV of global recovered cases

    Returns
    -------
    : pandas.DataFrame
    """
    # load & transform
    df_confirmed = _jhu_to_iso(fp_confirmed)
    df_deaths = _jhu_to_iso(fp_deaths)
    df_recovered = _jhu_to_iso(fp_recovered)

    # filter
    df = pd.DataFrame(columns=['date', 'confirmed', 'deaths', 'recovered']).set_index('date')
    df['confirmed'] = df_confirmed.loc[(country, state)]
    df['deaths'] = df_deaths.loc[(country, state)]
    df['recovered'] = df_recovered.loc[(country, state)]
    df.index.name = 'date'

    return df


def get_jhu_confirmed_cases():
    """
        Attempts to download the most current data from the online repository of the
        Coronavirus Visual Dashboard operated by the Johns Hopkins University
        and falls back to the backup provided with our repo if it fails.
        Only works if the module is located in the repo directory.

        Returns
        -------
        : confirmed_cases
            pandas table with confirmed cases
    """
    try:
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
        confirmed_cases = pd.read_csv(url, sep=",")
    except Exception as e:
        print("Failed to download current data, using local copy.")
        this_dir = os.path.dirname(__file__)
        confirmed_cases = pd.read_csv(
            this_dir + "/../data/confirmed_global_fallback_2020-04-28.csv", sep=","
        )

    return confirmed_cases


def filter_one_country(data_df, country, begin_date, end_date):
    """
    Returns the number of cases of one country as a np.array, given a dataframe returned by `get_jhu_confirmed_cases`
    Parameters
    ----------
    data_df : pandas.dataframe
    country : string
    begin_date : datetime.datetime
    end_date: datetime.datetime

    Returns
    -------
    : array
    """
    date_formatted_begin = _format_date(begin_date)
    date_formatted_end = _format_date(end_date)

    y = data_df[(data_df['Province/State'].isnull()) & (data_df['Country/Region']==country)]

    if len(y)==1:
        cases_obs = y.loc[:,date_formatted_begin:date_formatted_end]
    elif len(y)==0:
        cases_obs = data_df[data_df['Country/Region']==country].sum().loc[date_formatted_begin:date_formatted_end]

    else:
        raise RuntimeError('Country not found: {}'.format(country))

    return np.array(cases_obs).flatten()


def get_last_date(data_df):
    last_date = data_df.columns[-1]
    month, day, year = map(int, last_date.split("/"))
    return datetime.datetime(year + 2000, month, day)


