"""Functions for quick repetitive manual tasks to be called from the console for parex analysis. [call-time-model]
"""

import os
import json
import copy
import pprint
import numpy as np
import brian2 as b2
import matplotlib.pyplot as plt
from importlib import reload
from fcn import p_io, p_plot, p_util

# "global" variables
DEFAULT_PATH = '/extra/Seafile/ctm/out/param_exp_lo/'
PATH = DEFAULT_PATH


def setpath(pathname):
    """Sets global variable p_macro.PATH used in all other functions to the directory containing descriptive_stats.json

    :param pathname: full PATH to directory
    :type pathname: str
    :return:
    """

    global PATH

    assert os.path.isdir(pathname), 'ERROR: directory ' + pathname + ' was not found.'
    pathname = os.path.normpath(pathname) + os.path.sep
    if os.path.exists(pathname + "descriptive_stats.json"):
        PATH = pathname
        print("global PATH set to " + PATH)
    else:
        print("WARNING: pathname " + pathname + " doesn't contain file descriptive_stats.json. keeping original PATH "
              + PATH)


def parex2d(stat_x, stat_y, colorparam=None, i_colorparam=0, mark=None, identity_line=False, labels=True):
    """Mark all simulations with results in PATH in 2d space of the destrciptive stats (.json) given by stat_x & stat_y.

    :param stat_x: name of key in descriptive_stats.json to be mapped to the x-axis
    :type stat_x: str
    :param stat_y: name of key in descriptive_stats.json to be mapped to the y-axis
    :type stat_y: str
    :param colorparam: name of explored parameter, whose values should be used as a color code for dots/text
    :type colorparam: str
    :param i_colorparam: index to parameter given by colorparam (i.e. population/synapse index)
    :type i_colorparam: int
    :param mark: (list of) sim_id(s) to highlight on plot
    :type mark: int or list or None
    :param identity_line: if True, add an identity line (line through x==y)
    :type identity_line: bool
    :param labels: if True, add text labels of simulation ids next to points
    :type labels: bool
    :return: id_sims_all: list of ids of all simulations
    :rtype: id_sims_all: list
    """

    global PATH

    # check inputs
    if mark is not None and not isinstance(mark, list):
        mark = [mark]

    # load .json
    with open(PATH + "descriptive_stats.json", 'r') as file:
        stats = json.load(file)

    # create a list of colors for dots/text based on the value of each sim for one parameter
    if colorparam:
        # load config files and get a list combinations of the explored parameters for all simulations
        config_lo, _, _ = p_io.load_config(PATH + 'config_lo.json')
        config_hi, _, _ = p_io.load_config(PATH + 'config_hi.json')
        _, _, combinations, exp_param_names, all_values_per_param = \
            p_io.get_param_exploration_configs(config_lo, config_hi, b_generate_configs=False)

        assert colorparam in exp_param_names, "parameter " + colorparam + " is not one of the in explored parameters"
        idx_colorparam_in_combo_all = [i for i, v in enumerate(exp_param_names) if v == colorparam]
        idx_colorparam_in_combo = idx_colorparam_in_combo_all[i_colorparam]
        colorparam_values = [combinations[s][idx_colorparam_in_combo] for s in range(len(combinations))]
        unique_vals, unique_inverse = np.unique(colorparam_values, return_inverse=True)
        n_unique_vals = len(unique_vals)
        colormap = plt.cm.get_cmap('viridis', n_unique_vals+1)

    # set up figure
    plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 11})

    if identity_line:
        total_min = min([stats[stat_x], stats[stat_y]])
        total_max = max([stats[stat_x], stats[stat_y]])
        plt.plot([total_min, total_max], [total_min, total_max], color=[.5, .5, .5])

    # plot sim ids as text
    if labels:
        [plt.text(stats[stat_x][i], stats[stat_y][i], str(stats['id_sim'][i])) for i in range(len(stats['id_sim']))]

    # plot dots
    if colorparam:
        for unq_val in range(n_unique_vals):
            plt.plot([stats[stat_x][s] for s in range(len(stats['id_sim']))
                      if unique_inverse[stats['id_sim'][s]] == unq_val],
                     [stats[stat_y][s] for s in range(len(stats['id_sim']))
                      if unique_inverse[stats['id_sim'][s]] == unq_val],
                     '.', color=colormap(unq_val), label=unique_vals[unq_val], markersize=15, alpha=0.6)
        plt.legend(title=colorparam + ' [' + str(i_colorparam) + ']')
    else:
        plt.plot(stats[stat_x], stats[stat_y], '.k', markersize=15, alpha=0.6)

    if mark is not None and set(mark) & set(stats['id_sim']):
        i_mark = [int(np.where(np.array(stats['id_sim']) == m)[0]) for m in mark]
        for m in i_mark:
            if labels:
                plt.text(stats[stat_x][m], stats[stat_y][m], str(stats['id_sim'][m]),
                         bbox=dict(fc=(0, 1, 0), alpha=0.5))
            else:
                plt.plot(stats[stat_x], stats[stat_y], '.r')

    ax = plt.gca()
    ax.set_xlim([np.nanmin(stats[stat_x]) - 0.05 * np.abs(np.nanmin(stats[stat_x])),
                 np.nanmax(stats[stat_x]) + 0.05 * np.abs(np.nanmax(stats[stat_x]))])
    ax.set_ylim([np.nanmin(stats[stat_y]) - 0.05 * np.abs(np.nanmin(stats[stat_y])),
                 np.nanmax(stats[stat_y]) + 0.05 * np.abs(np.nanmax(stats[stat_y]))])
    ax.set_xlabel(stat_x)
    ax.set_ylabel(stat_y)

    return stats['id_sim']


def comborect(stat_x, stat_y, b_skip_fully_represented_params=True, fignum=None):
    """Select rectangular subspace in parex2d plot with two clicks (upper left and lower right corner) and print all
    parameter values in the combinations of simulations contained in that space. Returns ids of those simulations.

    :param stat_x: name of key in descriptive_stats.json mapped to the x-axis in the plot
    :type stat_x: str
    :param stat_y: name of key in descriptive_stats.json mapped to the y-axis in the plot
    :type stat_y: str
    :param b_skip_fully_represented_params: if True, don't print parameters for which all values are represented in rect
    :type b_skip_fully_represented_params: bool
    :param fignum: figure number of the figure containing the parex plot to be targeted. If None, will use plt.gcf()
    :type fignum: int or None
    :return: id_sims_in_rect: list of ids of the simulations that fall within the rectangle in 2d stats space
    :rtype: id_sims_in_rect: list
    """

    global PATH

    # check inputs
    if isinstance(fignum, int):
        if plt.fignum_exists(fignum):
            plt.figure(fignum)
        else:
            print('no figure with number ' + str(fignum) + ' found -> using gcf()')

    # load .json
    with open(PATH + "descriptive_stats.json", 'r') as file:
        stats = json.load(file)

    # get coordinates of top left and lower right rectangle corner
    coordinates = plt.ginput(2, timeout=0)
    x = np.sort((coordinates[0][0], coordinates[1][0]))
    y = np.sort((coordinates[1][1], coordinates[0][1]))
    print('x (' + stat_x + '): \t' + str(x) + '\ny (' + stat_y + '): \t' + str(y))

    # find simulations that fall within the rectangle in 2d stats space
    id_sims_in_rect = []
    for i in range(len(stats['id_sim'])):
        if x[0] <= stats[stat_x][i] <= x[1] and y[0] <= stats[stat_y][i] <= y[1]:
            id_sims_in_rect.append(stats['id_sim'][i])

    # create and print dictionary of all unique parameter values of simulations within the rectangle
    params(id_sims_in_rect, b_skip_fully_represented_params=b_skip_fully_represented_params)
    pprint.pprint('Simulations in rectangle: ' + str(id_sims_in_rect), compact=True)

    return id_sims_in_rect


def trace(id_sims, sortby=None, hline=None, xlims=None, figsize=(16, 9)):
    """plots trace of neuron of interest (as defined by config['plot']) of the parex simulation file in PATH,
    identified by id_sims (e.g. id_sims==23 => opens PATH/parex_000023.pkl)

    :param id_sims: (list of) id(s) of simulation(s) in parameter exploration, contained in .pkl file name
    :type id_sims: int or list
    :param sortby: name of descriptive stats parameter by which to sort the subplots, e.g. 't_spike1'
    :type sortby: str or None
    :param hline: y-value of gray horizontal line to add to each subplot
    :type hline: int or float
    :param xlims: x-axis limits (lower, upper)
    :type xlims: tuple or list
    :param figsize: size of figure in inches (width, height)
    :type figsize: tuple or list
    :return:
    """

    global PATH

    # check inputs
    if not isinstance(id_sims, list):
        id_sims = [id_sims]

    # load .json
    if sortby:
        with open(PATH + "descriptive_stats.json", 'r') as file:
            stats = json.load(file)

    n_traces = len(id_sims)
    n_gridcols = int(np.ceil(n_traces / 12))
    n_gridrows = int(np.ceil(n_traces / n_gridcols))
    plt.figure(figsize=figsize)
    plt.rcParams.update({'font.size': 11})

    # sort list of sims by a certain stat (if passed as argument)
    sims = copy.copy(id_sims)
    if sortby:
        if sortby in stats.keys():
            i_sims_oi = np.searchsorted(stats['id_sim'], sims)
            to_sort = [stats[sortby][i] for i in i_sims_oi]
            sims = [v for _, v in sorted(zip(to_sort, sims))]
        else:
            print(sortby + ' not found in descriptive stats -> traces not sorted!')

    # loop through sims and plot trace
    axs = []
    for i_sim, sim in enumerate(sims):
        filename = "parex_" + str(sim).zfill(6) + ".pkl"
        full_path = os.path.join(PATH, filename)
        if os.path.exists(full_path):
            # load simulation ouput
            states, spikes, _, _, _, config, info = p_io.load_monitors(full_path, b_silent=True)
            # get index to neuron of interest
            nrn_oi = p_util.get_abs_from_rel_nrn_idx(config['plot']['idx_nrn_oi_relative'],
                                                     config['plot']['idx_pop_oi'], config['misc']['n_neurons_per_pop'])
            axs.append(plt.subplot(n_gridrows, n_gridcols, i_sim + 1))
            # plot horizontal line
            if hline is not None:
                plt.axhline(hline, color=[.7, .7, .7], zorder=0)
            # plot trace
            plt.plot(states[0].t / b2.ms, states[0].v[nrn_oi, :] / b2.mV, color=[0, 0, 0], linewidth=2, label=str(sim))
            plt.legend(handlelength=0, framealpha=0)
            # plot artificial spikes
            spiketimes = spikes[0].t[spikes[0].i == nrn_oi] / b2.ms
            thresh = info[0]['v_thresh'][nrn_oi]
            if i_sim is not n_traces - 1:
                plt.gca().xaxis.set_ticklabels([])
            if i_sim is not 0:
                plt.gca().yaxis.set_ticklabels([])
            for s in range(len(spiketimes)):
                plt.plot([spiketimes[s], spiketimes[s]], [thresh, thresh + 30], color=[0, 0, 0], linewidth=2)
        else:
            print("FILE " + full_path + " NOT FOUND")

    # share y-axes
    [axs[n].get_shared_x_axes().join(axs[n], axs[n + 1]) for n in range(len(axs) - 1)]
    [axs[n].get_shared_y_axes().join(axs[n], axs[n + 1]) for n in range(len(axs) - 1)]
    if xlims:
        axs[0].set_xlim(xlims)
    else:
        axs[0].set_xlim([0, states[0].t[-1] / b2.ms])
    axs[0].autoscale(axis='y')


def params(id_sims, b_skip_fully_represented_params=True, b_combos=False):
    """Print values of the explored parameters in the combinations of simulations is_sims

    :param id_sims: (list of) integer id(s) of the simulations of interest
    :type id_sims: int or list
    :param b_skip_fully_represented_params: if True, don't print parameters for which all values are represented
        in id_sims
    :type b_skip_fully_represented_params: bool
    :param b_combos: if True, print explored parameter combinations of all sims
    :type b_combos: bool
    :return:
    """

    global PATH

    # check inputs
    if not isinstance(id_sims, list):
        id_sims = [id_sims]

    # load config files and get a list combinations of the explored parameters for all simulations
    config_lo, _, _ = p_io.load_config(PATH + 'config_lo.json')
    config_hi, _, _ = p_io.load_config(PATH + 'config_hi.json')
    _, _, combinations, exp_param_names, all_values_per_param = \
        p_io.get_param_exploration_configs(config_lo, config_hi, b_generate_configs=False)

    # get parameter value combinations of all simulations within the passed selection (id_sims)
    combinations_of_sims_oi = [combinations[i] for i in id_sims]
    if b_combos:
        with open(PATH + 'param_values.txt') as file:
            param_lines = file.readlines()
        param_lines_of_sims_oi = [param_lines[i] for i in id_sims]
        for line in param_lines_of_sims_oi:
            print(line, end='')

    # create and print dictionary of all unique parameter values of simulations within the selection
    unique_vals_in_selection = {}
    for i_par in range(len(combinations[0])):
        unique_vals_cur_par = np.unique([combinations_of_sims_oi[i][i_par]
                                         for i in range(len(combinations_of_sims_oi))])
        idx_values_of_param = np.where([p == exp_param_names[i_par] for p in exp_param_names])[0]
        nth_value_of_param = np.where(idx_values_of_param == i_par)[0]
        # first sublist in each dictionary key holds values of sims in selection, second holds all explored values
        all_explored_vals_cur_param = [round(val, 6) for val in all_values_per_param[i_par]]
        vals_in_selection = list(np.round(unique_vals_cur_par, 6))
        if set(vals_in_selection) == set(all_explored_vals_cur_param) and b_skip_fully_represented_params:
            continue
        unique_vals_in_selection[exp_param_names[i_par] + '_' + str(nth_value_of_param)] = \
            [vals_in_selection, all_explored_vals_cur_param]

    if b_skip_fully_represented_params:
        print('Parameters for which all values are represented in selection are not printed.')
    print("Selection of sims: " + str(id_sims))
    print("[[Parameter values of sims in selection], [All explored values of that parameter]]:")
    pprint.pprint(unique_vals_in_selection, compact=True)


def allsims():
    """Returns id_sims list of all simulations

    :return: id_sims_all: list of ids of all recorded simulations
    :rtype: id_sims_all: list
    """

    with open(PATH + "descriptive_stats.json", 'r') as file:
        stats = json.load(file)

    id_sims_all = stats['id_sim']

    return id_sims_all


def config(id_sims, b_save=False):
    """Print the config(s) of one or more parex simulations

    :param id_sims: (list of) integer id(s) of the simulations of interest
    :type id_sims: int or list
    :param b_save: if True, save config to .json file
    :type b_save: bool or int
    :return:
    """

    global PATH

    # check inputs
    if not isinstance(id_sims, list):
        id_sims = [id_sims]

    # load lo and hi config files and get a list of config dixtionaries of all simulations
    config_lo, _, _ = p_io.load_config(PATH + 'config_lo.json')
    config_hi, _, _ = p_io.load_config(PATH + 'config_hi.json')
    configs, _, _, _, _ = p_io.get_param_exploration_configs(config_lo, config_hi)

    # get only the configs of the relevant simulations and print them to console or .json file
    configs_of_sims_oi = [configs[i] for i in id_sims]
    for i, cfg in enumerate(configs_of_sims_oi):
        # save to .json file
        if b_save:
            run_id_str = os.path.normpath(PATH).split('_')[-1]
            filename = PATH + 'parex_cfg_' + run_id_str + '_' + str(id_sims[i]).zfill(6) + '.json'
            with open(filename, 'w') as file_cfg:
                json.dump(cfg, file_cfg, indent=4)
                print('Sim #: ' + str(id_sims[i]) + ' - config saved as ' + filename)
        else:
            print('Sim #: ' + str(id_sims[i]))
            pprint.pprint(cfg, compact=True)
