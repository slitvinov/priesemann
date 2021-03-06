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
rerun = True
date_data_begin = datetime.datetime(2020,3,1)
date_data_end = datetime.datetime(2020,4,21)
num_days_data = (date_data_end-date_data_begin).days
diff_data_sim = 16
num_days_future = 1
date_begin_sim = date_data_begin - datetime.timedelta(days = diff_data_sim)
date_end_sim   = date_data_end   + datetime.timedelta(days = num_days_future)
num_days_sim = (date_end_sim-date_begin_sim).days
prior_date_mild_dist_begin =  datetime.datetime(2020,3,9)
prior_date_strong_dist_begin =  datetime.datetime(2020,3,16)
prior_date_contact_ban_begin =  datetime.datetime(2020,3,23)
change_points = [dict(pr_mean_date_begin_transient = prior_date_mild_dist_begin,
                      pr_sigma_date_begin_transient = 3,
                      pr_median_lambda = 0.2,
                      pr_sigma_lambda = 0.5),
                 dict(pr_mean_date_begin_transient = prior_date_strong_dist_begin,
                      pr_sigma_date_begin_transient = 1,
                      pr_median_lambda = 1/8,
                      pr_sigma_lambda = 0.5),
                 dict(pr_mean_date_begin_transient = prior_date_contact_ban_begin,
                      pr_sigma_date_begin_transient = 1,
                      pr_median_lambda = 1/8/2,
                      pr_sigma_lambda = 0.5)]
if rerun:
    traces = []
    models = []
    for num_change_points in range(1, 4):
        model = cov19.SIR_with_change_points(new_cases_obs = np.diff(cases_obs),
                                            change_points_list = change_points[:num_change_points],
                                            date_begin_simulation = date_begin_sim,
                                            num_days_sim = num_days_sim,
                                            diff_data_sim = diff_data_sim,
                                            N = 83e6,
                                            priors_dict=None,
                                            weekends_modulated=True,
                                            weekend_modulation_type = 'abs_sine')
        models.append(model)
        traces.append(pm.sample(model=model, init='advi', draws=4000, tune=1000, cores = 12))
    pickle.dump([models, traces], open(path_save_pickled + 'a.pickled', 'wb'))
else:
    models, traces = pickle.load(open(path_save_pickled + 'a.pickled', 'rb'))
exec(open('figures_org.py').read())
create_figure_distributions(models[0], traces[0],
                              additional_insets = None, xlim_lambda = (0, 0.53), color = 'tab:red',
                              num_changepoints=1, xlim_tbegin=7, save_to = path_to_save +'distribution.1')
create_figure_distributions(models[1], traces[1],
                              additional_insets = None, xlim_lambda = (0, 0.53), color = 'tab:orange',
                              num_changepoints=2, xlim_tbegin=7, save_to = path_to_save +'distribution.2')
create_figure_distributions(models[2], traces[2],
                              additional_insets = None, xlim_lambda = (0, 0.53), color = 'tab:green',
                              num_changepoints=3, save_to = path_to_save + 'distribution.3')
create_figure_timeseries(traces[0], 'tab:red',
                         plot_red_axis=True, save_to=path_to_save + 'time.1', add_more_later = False)
create_figure_timeseries(traces[1], 'tab:orange',
                         plot_red_axis=True, save_to=path_to_save + 'time.2', add_more_later = False)
create_figure_timeseries(traces[2], 'tab:green',
                         plot_red_axis=True, save_to=path_to_save + 'time.3', add_more_later = False)
loo = [pm.loo(e, scale='deviance', pointwise=True) for e in traces]
for e in reversed(loo):
    print("lo: %.2f %.2f %.2f" % (e['loo'], e['loo_se'], e['p_loo']))
models[0].name = 'one point'
models[1].name = 'two points'
models[2].name = 'three points'
compare = pm.compare({models[0].name: traces[0],
                      models[1].name: traces[1],
                      models[2].name: traces[2]},
                     ic='LOO', scale='deviance')
print(compare)
