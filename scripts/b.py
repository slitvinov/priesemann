#!/usr/bin/env python3
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import pymc3 as pm
import scipy.stats
import sys
import theano
import theano.tensor as tt
import time as time_module

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append('../')
    import covid19_inference as cov19

path_to_save = '../figures/'
path_save_pickled = '../data/'
cases_obs = np.loadtxt("../data/germany.dat", dtype = int)
rerun = False
date_data_begin = datetime.datetime(2020,3,1)
date_data_end = datetime.datetime(2020,4,21)
num_days_data = (date_data_end-date_data_begin).days
diff_data_sim = 16 # should be significantly larger than the expected delay, in
                   # order to always fit the same number of data points.
num_days_future = 1
date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
date_end_sim   = date_data_end   + datetime.timedelta(days = num_days_future)
num_days_sim = (date_end_sim-date_begin_sim).days
prior_date_mild_dist_begin =  datetime.datetime(2020,3,9)
prior_date_strong_dist_begin =  datetime.datetime(2020,3,16)
prior_date_contact_ban_begin =  datetime.datetime(2020,3,23)

change_points = [dict(pr_mean_date_begin_transient = prior_date_strong_dist_begin,
                       pr_sigma_date_begin_transient = 14,
                       pr_median_transient_len=10,
                       pr_sigma_transient_len=5,
                       pr_median_lambda = 1/8,
                       pr_sigma_lambda = 1.0)]
if rerun:
    traces = []
    models = []
    model = cov19.SIR_with_change_points(new_cases_obs = np.diff(cases_obs),
                                        change_points_list = change_points,
                                        date_begin_simulation = date_begin_sim,
                                        num_days_sim = num_days_sim,
                                        diff_data_sim = diff_data_sim,
                                        N = 83e6,
                                        priors_dict=None,
                                        weekends_modulated=True,
                                        weekend_modulation_type = 'abs_sine')
    models.append(model)
    traces.append(pm.sample(model=model, init='advi', draws=4000, tune=1000, cores = 12))

    for i in range(len(models), 4):
        models.append(models[0])
        traces.append(traces[0])

    pickle.dump([models, traces], open(path_save_pickled + 'SIR_3scenarios_with_sine2.pickled', 'wb'))

else:
    models, traces = pickle.load(open(path_save_pickled + 'SIR_3scenarios_with_sine2.pickled', 'rb'))
exec(open('figures_revised.py').read())
trace = traces[3]
fig, ax = plt.subplots(figsize=(5,4))
time = np.arange(-len(cases_obs)+1, 0)
mpl_dates = conv_time_to_mpl_dates(time)
ax.plot(mpl_dates, np.abs(np.median(trace.new_cases[:, :num_days_data], axis=0) - np.diff(cases_obs)),
        'd', markersize=6,
         label='Absolute difference\n'
               'between fit and data')
ax.plot(mpl_dates, np.sqrt(np.median(trace.new_cases[:, :num_days_data], axis=0))*np.median(trace.sigma_obs, axis=0),
         label='Width of the likelihood', lw=3)
ax.set_ylabel('Difference (number of new cases)')
ax.set_xlabel('Date')
ax.legend(loc='upper left')
print(np.median(np.sum(trace.new_cases[:, :num_days_data], axis=1)+ trace.I_begin))
ax.xaxis.set_major_locator(matplotlib.dates.AutoDateLocator())
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%-m/%-d'))
create_figure_distributions(models[1], traces[1],
                              additional_insets = None, xlim_lambda = (0, 0.53), color = 'tab:red',
                              num_changepoints=1, xlim_tbegin=7, save_to = path_to_save + 'Comment')
create_figure_timeseries(traces[1], 'tab:red',
                         plot_red_axis=True, save_to=path_to_save + '1', add_more_later = False)
loo = [pm.loo(e) for e in traces]
for i in [1]:
    print(f"\nnumber of changepoints: {i}")
    for j in range(i+1):
        print(f'lambda* {j}')
        print(print_median_CI(traces[i][f"lambda_{j}"] - traces[i].mu, prec=2))
