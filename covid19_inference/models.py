import datetime
import theano
import theano.tensor as tt
import numpy as np
import pymc3 as pm

def SIR_with_change_points(
    new_cases_obs,
    change_points_list,
    date_begin_simulation,
    num_days_sim,
    diff_data_sim,
    N,
    priors_dict=None,
    weekends_modulated=False,
    weekend_modulation_type = 'step',
    student_nu = 4
):
    """
        Parameters
        ----------
        new_cases_obs : list or array
            Timeseries (day over day) of newly reported cases (not the total number)

        change_points_list : list of dicts
            List of dictionaries, each corresponding to one change point.

            Each dict can have the following key-value pairs. If a pair is not provided,
            the respective default is used.
                * pr_mean_date_begin_transient :     datetime.datetime, NO default
                * pr_median_lambda :                 number, same as default priors, below
                * pr_sigma_lambda :                  number, same as default priors, below
                * pr_sigma_date_begin_transient :    number, 3
                * pr_median_transient_len :          number, 3
                * pr_sigma_transient_len :           number, 0.3

        date_begin_simulation: datetime.datetime
            The begin of the simulation data

        num_days_sim : integer
            Number of days to forecast into the future

        diff_data_sim : integer
            Number of days that the simulation-begin predates the first data point in
            `new_cases_obs`. This is necessary so the model can fit the reporting delay.
            Set this parameter to a value larger than what you expect to find
            for the reporting delay.

        N : number
            The population size. For Germany, we used 83e6

        priors_dict : dict
            Dictionary of the prior assumptions

            Possible key-value pairs (and default values) are:
                * pr_beta_I_begin :        number, default = 100
                * pr_median_lambda_0 :     number, default = 0.4
                * pr_sigma_lambda_0 :      number, default = 0.5
                * pr_median_mu :           number, default = 1/8
                * pr_sigma_mu :            number, default = 0.2
                * pr_median_delay :        number, default = 8
                * pr_sigma_delay :         number, default = 0.2
                * pr_beta_sigma_obs :      number, default = 10
                * week_end_days :          tuple,  default = (6,7)
                * pr_mean_weekend_factor : number, default = 0.7
                * pr_sigma_weekend_factor :number, default = 0.17

        weekends_modulated : bool
            Whether to add the prior that cases are less reported on week ends. Multiplies the new cases numbers on weekends
            by a number between 0 and 1, given by a prior beta distribution. The beta distribution is parametrised
            by pr_mean_weekend_factor and pr_sigma_weekend_factor
        weekend_modulation_type : 'step' or 'abs_sine':
            whether the weekends are modulated by a step function, which only multiplies the days given by  week_end_days
            by the week_end_factor, or whether the whole week is modulated by an abs(sin(x)) function, with an offset
            with flat prior.
        Returns
        -------
        : pymc3.Model
            Returns an instance of pymc3 model with the change points

    """
    if priors_dict is None:
        priors_dict = dict()

    default_priors = dict(
        pr_beta_I_begin=100,
        pr_median_lambda_0=0.4,
        pr_sigma_lambda_0=0.5,
        pr_median_mu=1 / 8,
        pr_sigma_mu=0.2,
        pr_median_delay=8,
        pr_sigma_delay=0.2,
        pr_beta_sigma_obs=10,
        week_end_days = (6,7),
        pr_mean_weekend_factor=0.7,
        pr_sigma_weekend_factor=0.17
    )
    default_priors_change_points = dict(
        pr_median_lambda=default_priors["pr_median_lambda_0"],
        pr_sigma_lambda=default_priors["pr_sigma_lambda_0"],
        pr_sigma_date_begin_transient=3,
        pr_median_transient_len=3,
        pr_sigma_transient_len=0.3,
        pr_mean_date_begin_transient=None,
    )

    if not weekends_modulated:
        del default_priors['week_end_days']
        del default_priors['pr_mean_weekend_factor']
        del default_priors['pr_sigma_weekend_factor']

    for prior_name in priors_dict.keys():
        if prior_name not in default_priors:
            raise RuntimeError(f"Prior with name {prior_name} not known")
    for change_point in change_points_list:
        for prior_name in change_point.keys():
            if prior_name not in default_priors_change_points:
                raise RuntimeError(f"Prior with name {prior_name} not known")

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dict:
            priors_dict[prior_name] = value
            print(f"{prior_name} was set to default value {value}")
    for prior_name, value in default_priors_change_points.items():
        for i_cp, change_point in enumerate(change_points_list):
            if prior_name not in change_point:
                change_point[prior_name] = value
                print(
                    f"{prior_name} of change point {i_cp} was set to default value {value}"
                )

    if (
        diff_data_sim
        < priors_dict["pr_median_delay"]
        + 3 * priors_dict["pr_median_delay"] * priors_dict["pr_sigma_delay"]
    ):
        print("WARNING: diff_data_sim could be to small compared to the prior delay")
    if num_days_sim < len(new_cases_obs) + diff_data_sim:
        raise RuntimeError(
            "Simulation ends before the end of the data. Increase num_days_sim."
        )

    # ------------------------------------------------------------------------------ #
    # Model and prior implementation
    # ------------------------------------------------------------------------------ #

    with pm.Model() as model:
        # all pm functions now apply on the model instance
        # true cases at begin of loaded data but we do not know the real number
        I_begin = pm.HalfCauchy(name="I_begin", beta=priors_dict["pr_beta_I_begin"])

        # fraction of people that are newly infected each day
        lambda_list = []
        lambda_list.append(
            pm.Lognormal(
                name="lambda_0",
                mu=np.log(priors_dict["pr_median_lambda_0"]),
                sigma=priors_dict["pr_sigma_lambda_0"],
            )
        )
        for i, cp in enumerate(change_points_list):
            lambda_list.append(
                pm.Lognormal(
                    name=f"lambda_{i + 1}",
                    mu=np.log(cp["pr_median_lambda"]),
                    sigma=cp["pr_sigma_lambda"],
                )
            )

        # list of start dates of the transient periods of the change points
        tr_begin_list = []
        dt_before = date_begin_simulation
        for i, cp in enumerate(change_points_list):
            dt_begin_transient = cp["pr_mean_date_begin_transient"]
            if dt_before is not None and dt_before > dt_begin_transient:
                raise RuntimeError("Dates of change points are not temporally ordered")

            prior_mean = (
                dt_begin_transient - date_begin_simulation).days - 1  # convert the provided date format (argument) into days (a number)

            tr_begin = pm.Normal(
                name=f"transient_begin_{i}",
                mu=prior_mean,
                sigma=cp["pr_sigma_date_begin_transient"],
            )
            tr_begin_list.append(tr_begin)
            dt_before = dt_begin_transient

        # same for transient times
        tr_len_list = []
        for i, cp in enumerate(change_points_list):
            tr_len = pm.Lognormal(
                name=f"transient_len_{i}",
                mu=np.log(cp["pr_median_transient_len"]),
                sigma=cp["pr_sigma_transient_len"],
            )
            tr_len_list.append(tr_len)

        # build the time-dependent spreading rate
        lambda_t_list = [lambda_list[0] * tt.ones(num_days_sim)]
        lambda_before = lambda_list[0]

        for tr_begin, tr_len, lambda_after in zip(
            tr_begin_list, tr_len_list, lambda_list[1:]
        ):
            lambda_t = smooth_step_function(
                start_val=0,
                end_val=1,
                t_begin=tr_begin,
                t_end=tr_begin + tr_len,
                t_total=num_days_sim,
            ) * (lambda_after - lambda_before)
            lambda_before = lambda_after
            lambda_t_list.append(lambda_t)
        lambda_t = sum(lambda_t_list)

        # fraction of people that recover each day, recovery rate mu
        mu = pm.Lognormal(
            name="mu",
            mu=np.log(priors_dict["pr_median_mu"]),
            sigma=priors_dict["pr_sigma_mu"],
        )

        # delay in days between contracting the disease and being recorded
        delay = pm.Lognormal(
            name="delay",
            mu=np.log(priors_dict["pr_median_delay"]),
            sigma=priors_dict["pr_sigma_delay"],
        )

        # prior of the error of observed cases
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=priors_dict["pr_beta_sigma_obs"])

        # -------------------------------------------------------------------------- #
        # training the model with loaded data provided as argument
        # -------------------------------------------------------------------------- #

        S_begin = N - I_begin
        S, I, new_I = _SIR_model(
            lambda_t=lambda_t, mu=mu, S_begin=S_begin, I_begin=I_begin, N=N
        )

        new_cases_inferred = delay_cases(
            new_I_t=new_I,
            len_new_I_t=num_days_sim,
            len_out=num_days_sim - diff_data_sim,
            delay=delay,
            delay_diff=diff_data_sim,
        )

        if weekends_modulated:
            week_end_factor = pm.Beta('weekend_factor', mu=priors_dict['pr_mean_weekend_factor'],
                                                        sigma=priors_dict['pr_sigma_weekend_factor'])
            if weekend_modulation_type == 'step':
                modulation = np.zeros(num_days_sim - diff_data_sim)
                for i in range(num_days_sim - diff_data_sim):
                    date_curr = date_begin_simulation  + datetime.timedelta(days=i + diff_data_sim + 1)
                    if date_curr.isoweekday() in priors_dict['week_end_days']:
                        modulation[i] = 1
            elif weekend_modulation_type == 'abs_sine':
                offset_rad = pm.VonMises('offset_modulation_rad', mu = 0, kappa = 0.01)
                offset = pm.Deterministic('offset_modulation', offset_rad/(2*np.pi)*7)
                t = np.arange(num_days_sim - diff_data_sim)
                date_begin = date_begin_simulation + datetime.timedelta(days=diff_data_sim + 1)
                weekday_begin = date_begin.weekday()
                t -= weekday_begin # Sunday is zero
                modulation = 1-tt.abs_(tt.sin(t/7 * np.pi + offset_rad/2))

            multiplication_vec = np.ones(num_days_sim - diff_data_sim) - (1 - week_end_factor) * modulation
            new_cases_inferred_eff  = new_cases_inferred * multiplication_vec
        else:
            new_cases_inferred_eff = new_cases_inferred

        # likelihood of the model:
        # observed cases are distributed following studentT around the model.
        # we want to approximate a Poisson distribution of new cases.
        # we choose nu=4 to get heavy tails and robustness to outliers.
        # https://www.jstor.org/stable/2290063
        num_days_data = new_cases_obs.shape[-1]
        pm.StudentT(
            name="_new_cases_studentT",
            nu = student_nu,
            mu=new_cases_inferred_eff[:num_days_data],
            sigma=tt.abs_(new_cases_inferred[:num_days_data] + 1) ** 0.5
            * sigma_obs,  # +1 and tt.abs to avoid nans
            observed=new_cases_obs,
        )

        # add these observables to the model so we can extract a time series of them
        # later via e.g. `model.trace['lambda_t']`
        pm.Deterministic("lambda_t", lambda_t)
        pm.Deterministic("new_cases", new_cases_inferred_eff)
        pm.Deterministic("new_cases_raw", new_cases_inferred)
    return model



def _SIR_model(lambda_t, mu, S_begin, I_begin, N):
    """
        Implements the susceptible-infected-recovered model

        Parameters
        ----------
        lambda_t : ~numpy.ndarray
            time series of spreading rate, the length of the array sets the
            number of steps to run the model for

        mu : number
            recovery rate

        S_begin : number
            initial number of susceptible at first time step

        I_begin : number
            initial number of infected

        N : number
            population size

        Returns
        -------
        S : array
            time series of the susceptible

        I : array
            time series of the infected

        new_I : array
            time series of the new infected
    """

    new_I_0 = tt.zeros_like(I_begin)

    def next_day(lambda_t, S_t, I_t, _, mu, N):
        new_I_t = lambda_t / N * I_t * S_t
        S_t = S_t - new_I_t
        I_t = I_t + new_I_t - mu * I_t
        I_t = tt.clip(I_t, 0, N)  # for stability
        return S_t, I_t, new_I_t

    # theano scan returns two tuples, first one containing a time series of
    # what we give in outputs_info : S, I, new_I
    outputs, _ = theano.scan(
        fn=next_day,
        sequences=[lambda_t],
        outputs_info=[S_begin, I_begin, new_I_0],
        non_sequences=[mu, N],
    )
    return outputs

def delay_cases(new_I_t, len_new_I_t, len_out, delay, delay_diff):
    """
        Delays (time shifts) the input new_I_t by delay.

        Parameters
        ----------
        new_I_t : ~numpy.ndarray or theano vector
            Input to be delayed.

        len_new_I_t : integer
            Length of new_I_t. (Think len(new_I_t) ).
            Assure len_new_I_t is larger then len(new_cases_obs)-delay, otherwise it
            means that the simulated data is not long enough to be fitted to the data.

        len_out : integer
            Length of the output.

        delay : number
            If delay is an integer, the array will be exactly shifted. Else, the data
            will be shifted and intepolated (convolved with hat function of width one).
            Take care that delay is smaller than or equal to delay_diff,
            otherwise zeros are returned, which could potentially lead to errors.

        delay_diff: integer
            The difference in length between the new_I_t and the output.

        Returns
        -------
            an array with length len_out that was time-shifted by delay
    """

    # elementwise delay of input to output
    delay_mat = make_delay_matrix(
        n_rows=len_new_I_t, n_columns=len_out, initial_delay=delay_diff
    )
    inferred_cases = interpolate(new_I_t, delay, delay_mat)
    return inferred_cases


def make_delay_matrix(n_rows, n_columns, initial_delay=0):
    """
        Has in each entry the delay between the input with size n_rows and the output
        with size n_columns

        initial_delay is the top-left element.
    """
    size = max(n_rows, n_columns)
    mat = np.zeros((size, size))
    for i in range(size):
        diagonal = np.ones(size - i) * (initial_delay + i)
        mat += np.diag(diagonal, i)
    for i in range(1, size):
        diagonal = np.ones(size - i) * (initial_delay - i)
        mat += np.diag(diagonal, -i)
    return mat[:n_rows, :n_columns]

def interpolate(array, delay, delay_matrix):
    """
        smooth the array (if delay is no integer)
    """
    interp_matrix = tt.maximum(1 - tt.abs_(delay_matrix - delay), 0)
    interpolation = tt.dot(array, interp_matrix)
    return interpolation


def smooth_step_function(start_val, end_val, t_begin, t_end, t_total):
    """
        Instead of going from start_val to end_val in one step, make the change a
        gradual linear slope.

        Parameters
        ----------
            start_val : number
                Starting value

            end_val : number
                Target value

            t_begin : number or array (theano)
                Time point (inbetween 0 and t_total) where start_val is placed

            t_end : number or array (theano)
                Time point (inbetween 0 and t_total) where end_val is placed

            t_total : integer
                Total number of time points

        Returns
        -------
            : theano vector
                vector of length t_total with the values of the parameterised f(t)
    """
    t = np.arange(t_total)

    return (
        tt.clip((t - t_begin) / (t_end - t_begin), 0, 1) * (end_val - start_val)
        + start_val
    )
