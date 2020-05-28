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

def get_jhu_confirmed_cases():
    """
        Returns
        -------
        : confirmed_cases
            pandas table with confirmed cases
    """
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

_format_date = lambda date_py: "{}/{}/{}".format(
    date_py.month, date_py.day, str(date_py.year)[2:4]
)
