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
                       pr_sigma_date_begin_transient = 1,
                       pr_median_transient_len=16,
                       pr_sigma_transient_len=5,
                       pr_median_lambda = 1/8,
                       pr_sigma_lambda = 0.5)]
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
    pickle.dump([models, traces], open(path_save_pickled + 'b.pickle', 'wb'))

else:
    models, traces = pickle.load(open(path_save_pickled + 'b.pickle', 'rb'))
exec(open('figures_org.py').read())
create_figure_distributions(models[0], traces[0],
                            additional_insets = None, xlim_lambda = (0, 0.53), color = 'tab:red',
                            num_changepoints=1, xlim_tbegin=7, save_to = path_to_save + 'distribution.1b')
create_figure_timeseries(traces[0], 'tab:red',
                         plot_red_axis=True, save_to=path_to_save + 'time.1b', add_more_later = False)
loo = [pm.loo(e) for e in traces]
for e in loo:
    print("lo: %.1f %.1f" % (-2*e['loo'], 2*e['loo_se']))
