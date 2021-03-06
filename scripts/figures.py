from matplotlib import rc_context
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
import locale
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pymc3 as pm
import scipy.stats
import sys
import theano
import time as time_module

try:
    import covid19_inference as cov19
except ModuleNotFoundError:
    sys.path.append("../..")
    import covid19_inference as cov19


# ------------------------------------------------------------------------------ #
# global settings and variables
# ------------------------------------------------------------------------------ #

# styling for prior distributions
prio_style = {
    "color": "#708090",
    "linewidth": 3,
    "label": "Prior",
}
# styling for posterior distributions
post_style = {
    "density": True,
    "color": "tab:orange",
    "label": "Posterior",
    "zorder": -2,
}

# format the date on the x axis (see https://strftime.org/) example April 1 2020
date_format = "%b %-d"  # Apr 1
try:
    locale.setlocale(locale.LC_ALL, "en_US")
except:
    pass

# whether to show the minor ticks (for every day)
date_show_minor_ticks = True

# set to None to keep everything a vector, with `-1` Posteriors are rastered (see above)
rasterization_zorder = -1

def create_figure_accuracy(y, color, save_to):
    num_days_futu_to_plot=1
    ylabel_new = f"pointwise predictive accuracy"
    new_c_ylim = [0, 10_000]
    fig, axes = plt.subplots(1, 1, gridspec_kw={"height_ratios": [1]}, constrained_layout=True)
    diff_to_0 = num_days_data + diff_data_sim
    start_date = conv_time_to_mpl_dates(-len(y[0]) + 1) + diff_to_0
    end_date = conv_time_to_mpl_dates(num_days_futu_to_plot) + diff_to_0
    mid_date = conv_time_to_mpl_dates(1) + diff_to_0
    time_past = np.arange(-len(y[0]), 1)
    time_futu = np.arange(0, num_days_futu_to_plot + 1)
    mpl_dates_past = conv_time_to_mpl_dates(time_past) + diff_to_0
    ax = axes
    for y0, c0 in zip(y, color):
        ax.plot(mpl_dates_past[1:], y0, "-", linewidth=3, color=c0)
    ax.set_ylabel(ylabel_new)
    ax.set_rasterization_zorder(rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)
    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)
    plt.savefig(save_to + ".png", dpi=100, bbox_inches="tight", pad_inches=0.05)
    return fig, axes


def create_figure_timeseries(
    trace,
    color="tab:green",
    save_to=None,
    num_days_futu_to_plot=1,
    plot_red_axis=True,
    axes=None,
    forecast_label="Forecast",
    add_more_later=False,
):
    plot_par = dict()
    plot_par["draw_ci_95"] = True
    plot_par["draw_ci_75"] = False

    ylabel_new = f"new cases"

    pos_letter = (-0.3, 1)
    new_c_ylim = [0, 10_000]

    cum_c_ylim = [0, 300_000]
    cum_c_insetylim = [50, 300_000]

    color_futu = color
    color_past = color
    fig, axes = plt.subplots(
        1,
        1,
        gridspec_kw={"height_ratios": [1]},
        constrained_layout=True,
    )
    if add_more_later:
        color_past = "#646464"


    diff_to_0 = num_days_data + diff_data_sim

    # interval for the plots with forecast
    start_date = conv_time_to_mpl_dates(-len(cases_obs) + 2) + diff_to_0
    end_date = conv_time_to_mpl_dates(num_days_futu_to_plot) + diff_to_0
    mid_date = conv_time_to_mpl_dates(1) + diff_to_0

    # x-axis for dates, new_cases are one element shorter than cum_cases, use [1:]
    # 0 is the last recorded data point
    time_past = np.arange(-len(cases_obs) + 1, 1)
    time_futu = np.arange(0, num_days_futu_to_plot + 1)
    mpl_dates_past = conv_time_to_mpl_dates(time_past) + diff_to_0
    mpl_dates_futu = conv_time_to_mpl_dates(time_futu) + diff_to_0

    # --------------------------------------------------------------------------- #
    # prepare data
    # --------------------------------------------------------------------------- #
    # observed data, only one dim: [day]
    new_c_obsd = np.diff(cases_obs)
    cum_c_obsd = cases_obs

    # model traces, dims: [sample, day],
    new_c_past = trace["new_cases"][:, :num_days_data]
    new_c_futu = trace["new_cases"][
        :, num_days_data : num_days_data + num_days_futu_to_plot
    ]
    cum_c_past = np.cumsum(np.insert(new_c_past, 0, 0, axis=1), axis=1) + cases_obs[0]
    cum_c_futu = np.cumsum(np.insert(new_c_futu, 0, 0, axis=1), axis=1) + cases_obs[-1]

    # --------------------------------------------------------------------------- #
    # New cases, lin scale first
    # --------------------------------------------------------------------------- #
    ax = axes
    ax.plot(
        mpl_dates_past[1:],
        new_c_obsd,
        "o",
        label="Data",
        markersize=12,
        color="tab:blue",
        zorder=5,
    )
    ax.plot(
        mpl_dates_past[1:],
        np.median(new_c_past, axis=0),
        "-",
        color=color_past,
        linewidth=4.5,
        label="Fit",
        zorder=10,
    )
    if plot_par["draw_ci_95"] == True:
        ax.fill_between(
            mpl_dates_past[1:],
            np.percentile(new_c_past, q=2.5, axis=0),
            np.percentile(new_c_past, q=97.5, axis=0),
            alpha=0.1,
            color=color_past,
            lw=0,
        )

    ax.set_ylabel(ylabel_new)
    ax.set_ylim(new_c_ylim)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(format_k))

    # --------------------------------------------------------------------------- #
    # Finalize
    # --------------------------------------------------------------------------- #

    ax.set_rasterization_zorder(rasterization_zorder)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlim(start_date, end_date)
    format_date_xticks(ax)

    for label in ax.xaxis.get_ticklabels()[1::2]:
        label.set_visible(False)

    if save_to is not None:
        plt.savefig(
            save_to + ".png", dpi=150, bbox_inches="tight", pad_inches=0.05,
        )
    return fig, axes

def get_priors_dict():
    pr = dict()
    pr["lambda_0"] = lambda x: scipy.stats.lognorm.pdf(x, scale=0.4, s=0.5)
    pr["lambda_1"] = lambda x: scipy.stats.lognorm.pdf(x, scale=0.2, s=0.5)
    pr["lambda_2"] = lambda x: scipy.stats.lognorm.pdf(x, scale=1 / 8, s=0.5)
    pr["lambda_3"] = lambda x: scipy.stats.lognorm.pdf(x, scale=1 / 8 / 2, s=0.5)
    pr["transient_begin_0"] = lambda x: scipy.stats.norm.pdf(
        x, loc=matplotlib.dates.date2num([prior_date_mild_dist_begin])[0], scale=3
    )
    pr["transient_begin_1"] = lambda x: scipy.stats.norm.pdf(
        x, loc=matplotlib.dates.date2num([prior_date_strong_dist_begin])[0], scale=1
    )
    pr["transient_begin_2"] = lambda x: scipy.stats.norm.pdf(
        x, loc=matplotlib.dates.date2num([prior_date_contact_ban_begin])[0], scale=1
    )
    pr["transient_len_0"] = lambda x: scipy.stats.lognorm.pdf(x, scale=3, s=0.3)
    pr["transient_len_1"] = lambda x: scipy.stats.lognorm.pdf(x, scale=3, s=0.3)
    pr["transient_len_2"] = lambda x: scipy.stats.lognorm.pdf(x, scale=3, s=0.3)
    pr["mu"] = lambda x: scipy.stats.lognorm.pdf(x, scale=1 / 8, s=0.2)
    pr["delay"] = lambda x: scipy.stats.lognorm.pdf(x, scale=8, s=0.2)
    pr["I_begin"] = lambda x: scipy.stats.halfcauchy.pdf(x, scale=100)
    pr["sigma_obs"] = lambda x: scipy.stats.halfcauchy.pdf(x, scale=10)
    return pr


def get_label_dict():
    labels = dict()
    labels["mu"] = f"Recovery rate"
    labels["delay"] = f"Reporting delay"
    labels["I_begin"] = f"Initial infections"
    labels["sigma_obs"] = f"Scale (width)\nof the likelihood"
    labels["lambda_0"] = f"Initial rate"
    labels["lambda_1"] = f"Spreading rates"
    labels["transient_begin_0"] = f"Change times"
    labels["transient_len_0"] = f"Change duration"
    labels["E_begin_scale"] = "Initial scale\nof exposed"
    labels["median_incubation"] = "Median\nincubation delay"
    labels["sigma_random_walk"] = "Std. of\nrandom walk"
    labels["weekend_factor"] = "Weekend, amplitude\nof modulation"
    labels["offset_modulation_rad"] = "Weekend, offset\nfrom sunday"
    return labels


def get_mpl_text_coordinates(text, ax):
    """
        helper to get a coordinates of a text object in the coordinates of the
        axes element.
        used for the rectangle backdrop.

        Returns:
        x_min, x_max, y_min, y_max
    """
    fig = ax.get_figure()

    try:
        fig.canvas.renderer
    except Exception as e:
        # print(e)
        # otherwise no renderer, needed for text position calculation
        fig.canvas.draw()

    x_min = None
    x_max = None
    y_min = None
    y_max = None

    # get bounding box of text
    transform = ax.transAxes.inverted()
    try:
        bb = text.get_window_extent(renderer=fig.canvas.get_renderer())
    except:
        bb = text.get_window_extent()
    bb = bb.transformed(transform)
    x_min = bb.get_points()[0][0]
    x_max = bb.get_points()[1][0]
    y_min = bb.get_points()[0][1]
    y_max = bb.get_points()[1][1]

    return x_min, x_max, y_min, y_max


def add_mpl_rect_around_text(text_list, ax, **kwargs):
    """
        add a rectangle to the axes (behind the text)

        provide a list of text elements and possible options passed to patches.Rectangle

        e.g.
        facecolor="grey",
        alpha=0.2,
        zorder=99,

    """

    x_gmin = 1
    y_gmin = 1
    x_gmax = 0
    y_gmax = 0

    for text in text_list:
        x_min, x_max, y_min, y_max = get_mpl_text_coordinates(text, ax)
        if x_min < x_gmin:
            x_gmin = x_min
        if y_min < y_gmin:
            y_gmin = y_min
        if x_max > x_gmax:
            x_gmax = x_max
        if y_max > y_gmax:
            y_gmax = y_max

    # coords between 0 and 1 (relative to axes) add 10% margin
    y_gmin = np.clip(y_gmin - 0.15, 0, 1)
    y_gmax = np.clip(y_gmax + 0.15, 0, 1)
    # x_gmin *= 1 - pad_percent / 100
    # x_gmax *= 1 + pad_percent / 100

    rect = patches.Rectangle(
        (x_gmin, y_gmin),
        x_gmax - x_gmin,
        y_gmax - y_gmin,
        transform=ax.transAxes,
        **kwargs,
    )

    ax.add_patch(rect)


def create_figure_distributions(
    model,
    trace,
    save_to=None,
    additional_insets=None,
    xlim_lambda=(0, 0.53),
    color="tab:green",
    num_changepoints=3,
    xlim_tbegin=4,
    trace_prior=None,
    name_trans_day="transient_begin_",
):
    """
        create the distribution overview plot. only using layout 2 from now on.

        3 columns by default. if additional insets: create fourch column (make wider
        figure)
    """
    if additional_insets is None:
        additional_insets = {}
    colors = ["#708090", color]

    additional_dists = len(additional_insets)
    num_rows = num_changepoints + 3
    num_columns = 3 + int(np.ceil(additional_dists / num_rows))
    width_col = 4.5 / 3 * num_columns
    height_fig = 6 if not num_changepoints == 1 else 5
    fig, axes = plt.subplots(
        num_rows, num_columns, figsize=(width_col, 6), constrained_layout=True
    )

    xlim_transt = (0, 7)
    xlim_tbegin = xlim_tbegin  # median plus minus x days

    axpos = dict()
    letters = dict()

    # text prefix for insets (meidan, ci).
    # leave away the closing doller, we add it later
    insets = dict()
    insets["lambda_0"] = r"$\lambda_0 \simeq "
    for i in range(0, num_changepoints):
        insets[f"lambda_{i+1}"] = f"$\lambda_{i+1} \simeq "
        insets[f"transient_begin_{i}"] = f"$t_{i+1} \simeq "
        insets[f"transient_len_{i}"] = f"$\Delta t_{i+1} \simeq "

    insets["mu"] = r"$\mu \simeq "
    insets["delay"] = r"$D \simeq "
    insets["sigma_obs"] = r"$\sigma \simeq "
    insets["I_begin"] = r"$I_0 \simeq "
    insets["weekend_factor"] = r"$f_w \simeq "
    insets["offset_modulation_rad"] = r"$\phi_{w} \simeq "

    for key, inset in additional_insets.items():
        insets[key] = inset

    # layout 2
    pos_letter = (-0.4, 1.1)
    labels = get_label_dict()
    axpos["lambda_0"] = axes[2][0]
    for i_cp in range(0, num_changepoints):
        axpos["lambda_{}".format(i_cp + 1)] = axes[i_cp + 3][0]
        axpos["transient_begin_{}".format(i_cp)] = axes[i_cp + 3][1]
        axpos["transient_len_{}".format(i_cp)] = axes[i_cp + 3][2]

    axpos["mu"] = axes[1][0]
    axpos["delay"] = axes[2][2]
    axpos["I_begin"] = axes[2][1]
    axpos["sigma_obs"] = axes[1][1]
    axpos["weekend_factor"] = axes[0][0]
    axpos["offset_modulation_rad"] = axes[0][1]

    # this is new, stretching legend over two panels
    for ax in axes[0:2, 2]:
        ax.remove()
    gs = axes[0, 0].get_gridspec()
    axpos["legend"] = fig.add_subplot(gs[0:2, 2])
    # axpos["legend"] = axes[1][2]

    letters["weekend_factor"] = r"D"
    letters["mu"] = r"E"
    letters["lambda_0"] = r"F"
    letters["lambda_1"] = r"G"

    i = num_rows - 1
    for i, (key, inset) in enumerate(additional_insets.items()):
        print(f"additional insets: {key}")
        col = 3
        row = i % num_rows
        axpos[key] = axes[row][col]
    # for i in range(i + 1, num_rows):
    #     axes[i][col].set_visible(False)

    # if not len(additional_insets) % num_rows == 1:
    # axpos["legend"] = axes[0][num_columns - 1]

    # render panels
    for key in axpos.keys():
        if "legend" in key:
            continue

        data = trace[key]
        if "transient_begin" in key:
            data = conv_time_to_mpl_dates(trace[key])
        elif "weekend_factor_rad" == key:
            data = data / np.pi / 2 * 7

        ax = axpos[key]
        # using .get returns None when the key is not in the dict, and avoids KeyError
        ax.set_xlabel(labels.get(key))
        ax.xaxis.set_label_position("top")

        # make some bold
        if key == "lambda_1" or key == "transient_begin_0" or key == "transient_len_0":
            ax.set_xlabel(labels.get(key), fontweight="bold")

        # posteriors
        ax.hist(
            data,
            bins=50,
            density=True,
            color=colors[1],
            label="Posterior",
            alpha=0.7,
            zorder=-5,
        )

        # xlim
        if "lambda" in key or "mu" == key:
            ax.set_xlim(xlim_lambda)
            ax.axvline(np.median(trace["mu"]), ls=":", color="black")
        elif "I_begin" == key:
            ax.set_xlim(0)
        elif "transient_len" in key:
            ax.set_xlim(xlim_transt)
        elif "transient_begin" in key:
            md = np.median(data)
            ax.set_xlim([int(md) - xlim_tbegin, int(md) + xlim_tbegin - 1])
            format_date_xticks(ax)

        # priors
        limits = ax.get_xlim()
        x_for_ax = np.linspace(*limits, num=100)
        x_for_pr = x_for_ax
        if "transient_begin" in key:
            beg_x = matplotlib.dates.num2date(x_for_ax[0])
            diff_dates_x = (beg_x.replace(tzinfo=None) - date_begin_sim).days
            x_for_pr = x_for_ax - x_for_ax[0] + diff_dates_x
        if "weekend_factor_rad" == key:
            x_for_ax *= np.pi * 2 / 7

        if trace_prior is None:
            prior_dist = get_prior_distribution(model, x_for_pr, key)
        else:
            kde = scipy.stats.gaussian_kde(trace_prior[key])
            prior_dist = kde.evaluate((x_for_pr))

        ax.plot(
            x_for_ax, prior_dist, label="Prior", color=colors[0], linewidth=3,
        )
        ax.set_xlim(*limits)

        # letters
        if key in letters.keys():
            ax.text(
                pos_letter[0],
                pos_letter[1],
                letters[key],
                transform=ax.transAxes,
                size=14,
                horizontalalignment="left",
            )

        # median
        global text
        if "lambda" in key or "mu" == key or "sigma_random_walk" == key:
            text = print_median_CI(data, prec=2)
        elif "transient_begin" in key:
            text = print_median_CI(
                data - matplotlib.dates.date2num(date_data_begin) + 1, prec=1
            )
        else:
            text = print_median_CI(data, prec=1)

        if key in insets.keys():
            # strip everything except the median value
            text = text.replace("Median: ", "").replace("CI: ", "")
            md = text.split("\n")[0]
            ci = text.split("\n")[1]

            # create the inset text and we want a bounding box around the compound
            # text = insets[key] + md + "$" + "\n" + r"$\,$"
            text = insets[key] + md + "$"
            t_xl = ax.text(
                0.6,
                0.9,
                text,
                fontsize=12,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                zorder=100,
            )

            x_min, x_max, y_min, y_max = get_mpl_text_coordinates(t_xl, ax)

            t_sm = ax.text(
                0.6,
                y_min * 0.9,  # let's have a ten perecent margin or so
                ci,
                fontsize=9,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                # bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
                zorder=101,
            )

            add_mpl_rect_around_text(
                [t_xl, t_sm], ax, facecolor="white", alpha=0.5, zorder=99,
            )

    # legend
    if "legend" in axpos:
        ax = axpos["legend"]
        ax.set_axis_off()
        ax.plot([], [], color=colors[0], linewidth=3, label="Prior")
        ax.hist([], color=colors[1], label="Posterior")
        ax.legend(loc="center")
        ax.get_legend().get_frame().set_linewidth(0.0)
        ax.get_legend().get_frame().set_facecolor("#F0F0F0")
        ax.get_legend().set_title(
            f"Data until\n{date_data_end.strftime('%B %-d')}", prop=dict(weight="bold")
        )

    # dont draw empty panels
    for ax in axes.flatten():
        if ax not in axpos.values():
            ax.set_visible(False)

    # dirty hack to get some space at the bottom to align with timeseries
    if not num_changepoints == 1:
        axes[-1][0].xaxis.set_label_position("bottom")
        axes[-1][0].set_xlabel(r"$\,$")

    for jdx, ax_row in enumerate(axes):
        for idx, ax in enumerate(ax_row):
            if idx == 0 and jdx == num_rows - 1:
                ax.set_ylabel("Density")
            ax.tick_params(labelleft=False)
            ax.locator_params(nbins=4)
            ax.set_rasterization_zorder(rasterization_zorder)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

    # plt.subplots_adjust(wspace=0.2, hspace=0.9)

    if save_to is not None:
        plt.savefig(save_to + ".png", bbox_inches="tight", pad_inches=0.05, dpi=300)

# format yaxis 10_000 as 10 k
format_k = lambda num, _: "${:.0f}\,$k".format(num / 1_000)

# format xaxis, ticks and labels
def format_date_xticks(ax, minor=None):
    ax.xaxis.set_major_locator(
        matplotlib.dates.WeekdayLocator(interval=1, byweekday=matplotlib.dates.SU)
    )
    if minor is None:
        minor = date_show_minor_ticks
    if minor:
        ax.xaxis.set_minor_locator(matplotlib.dates.DayLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter(date_format))


def truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)


def print_median_CI(arr, prec=2):
    f_trunc = lambda n: truncate_number(n, prec)
    med = f_trunc(np.median(arr))
    perc1, perc2 = (
        f_trunc(np.percentile(arr, q=2.5)),
        f_trunc(np.percentile(arr, q=97.5)),
    )
    return "Median: {}\nCI: [{}, {}]".format(med, perc1, perc2)


def conv_time_to_mpl_dates(arr):
    try:
        return matplotlib.dates.date2num(
            [datetime.timedelta(days=float(date)) + date_begin_sim for date in arr]
        )
    except:
        return matplotlib.dates.date2num(
            datetime.timedelta(days=float(arr)) + date_begin_sim
        )

def get_prior_distribution(model, x, varname):
    """
    Given a model and variable name, returns the prior distribution evaluated at x.
    Parameters
    ----------
    model: pm.Model instance
    x: list or array
    varname: string

    Returns
    -------
    : array
    """
    return np.exp(model[varname].distribution.logp(x).eval())
