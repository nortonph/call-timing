"""Various functions for plotting model properties and results, as well as recorded data. [call-time-model]
"""

import numpy as np
from scipy.signal import savgol_filter
import brian2 as b2
import matplotlib.pyplot as plt
import matplotlib.gridspec
from types import SimpleNamespace

# import my functions
import fcn.p_util
from fcn import p_analysis, p_util

# color palette adapted from http://mkweb.bcgsc.ca/biovis2012/ (colorblind-friendly)
C_COLORS = [[0.000, 0.427, 0.859],  # 0 blue
            [0.859, 0.427, 0.000],  # 1 orange
            [0.714, 0.427, 1.000],  # 2 grape
            [0.573, 0.000, 0.000],  # 3 red
            [0.845, 0.798, 0.171],  # 4 yellow
            [0.000, 0.286, 0.286],  # 5 teal
            [0.286, 0.000, 0.573],  # 6 purple
            [0.000,   0.0,   0.0]]  # -1 black
C_COLORS_LIGHT = [[0.200, 0.627, 1.000],  # 0 blue
                  [1.000, 0.627, 0.200],  # 1 orange
                  [0.914, 0.627, 1.000],  # 2 grape
                  [0.773, 0.200, 0.200],  # 3 red
                  [0.945, 0.898, 0.271],  # 4 yellow
                  [0.200, 0.486, 0.486],  # 5 teal
                  [0.486, 0.200, 0.773],  # 6 purple
                  [0.000,   0.0,   0.0]]  # -1 black
C_COLORS_GREY = [[0.0, 0.0, 0.0],  # 0 black
                 [0.1, 0.1, 0.1],
                 [0.2, 0.2, 0.2],
                 [0.3, 0.3, 0.3],
                 [0.4, 0.4, 0.4],
                 [0.5, 0.5, 0.5],
                 [0.6, 0.6, 0.6],
                 [0.7, 0.7, 0.7],
                 [0.8, 0.8, 0.8],
                 [0.9, 0.9, 0.9],
                 [1.0, 1.0, 1.0]]  # 10 white
CMAP_NAME_CONTINUOUS = 'viridis'


def get_color_list(n_colors, color_map_in=None, i_colors=None):
    """Get a list of 3-element lists containing RGB values (0-1) of n_colors length

    :param n_colors: number of list entries (colors) needed
    :type n_colors: int
    :param color_map_in: [default=None] color map (list of 3-element lists) that should be repeated to get n_colors
    :type color_map_in: [[float, float, float]] or None
    :param i_colors: [default=None] list of indices to colors of the colormap (i.e. order of colors) of length n_colors
    :type i_colors: list or None
    :return: color_map_out: list of 3-element lists containing RGB values (0-1) of (at least) n_colors length
    :rtype: [[float, float, float]]
    """

    if not color_map_in:
        # use default color map
        color_map_in = C_COLORS
    if not i_colors:
        if n_colors <= len(color_map_in):
            i_colors = list(range(n_colors))
        else:
            i_colors = list(range(len(color_map_in)))

    # repeat color map as many times as necessary to get n_colors
    color_map_out = []
    for rep in range(np.ceil(n_colors / len(color_map_in)).astype(int)):
        color_map_out = color_map_out + [color_map_in[i] for i in i_colors]

    return color_map_out


def plot_colormap_values(cmap):
    """ Plot RGB values of a colormap across the full range.

    :param cmap: colormap as returned by plt.cm.get_cmap()
    :return:
    """

    # get 100 colormap values
    cm = np.array([cmap(i) for i in np.arange(0, 1, 0.01)])

    # extract and plot RGB values
    r = [cm[i][0] for i in range(len(cm))]
    g = [cm[i][1] for i in range(len(cm))]
    b = [cm[i][2] for i in range(len(cm))]
    plt.plot(r, '-r')
    plt.plot(g, '-g')
    plt.plot(b, '-b')

    # plot color bar above
    [plt.plot((i - .5, i + .5), (1.1, 1.1), color=cm[i], linewidth=10) for i in range(len(cm))]


def format_fig(handle_fig, handle_ax, fig_size_inches=None, x_label=None, y_label=None,
               b_remove_box=False, b_remove_legend=False, b_make_ticklabel_font_sansserif=False):
    """Format figures/axes after plotting.

    :param handle_fig: (list of) handle(s) of figures to be formatted
    :type handle_fig: matplotlib.figure.Figure
    :param handle_ax: (list of) handle(s) of axes to be formatted
    :type handle_ax: matplotlib.axes._subplots.AxesSubplot
    :param fig_size_inches: tuple (x, y) determining the size of the figure
    :type fig_size_inches: tuple or list
    :param x_label: new x-axis label
    :type x_label: str
    :param y_label: new y-axis label
    :type y_label: str
    :param b_remove_box: remove the top and right frame lines while keeping the axes intact
    :type b_remove_box: bool or int
    :param b_remove_legend: self explanatory
    :type b_remove_legend: bool or int
    :param b_make_ticklabel_font_sansserif: make ticklabels font sans-serif (maybe necessary with latex text formatting)
        for technical reasons, ticklabels get removed if there is no axis label. to keep ticklabels in this case, set
        the axis label to a whitespce beforehand, e.g.: ax.set_xlabel(' ')
    :type b_make_ticklabel_font_sansserif: bool or int
    :return:
    """

    # if figure and axis handles are single elements, convert them to lists
    if not isinstance(handle_fig, list):
        handle_fig = [handle_fig]
    if not isinstance(handle_ax, list):
        handle_ax = [handle_ax]

    # change figure properties
    for fig in handle_fig:
        if fig_size_inches:
            fig.set_size_inches(fig_size_inches)

    # change axis properties
    for ax in handle_ax:
        if b_remove_legend:
            ax.get_legend().remove()
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)
        if b_remove_box:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    # make ticklabels have sans serif fonts (might be necessary if using latex text formatting). extracts tick vlaues,
    # convert to int if w/o decimals, convert to strings and replace ASCII minus sign with Unicode one (long dash).
    # ticklabels get removed if the is no axis label. to keep ticklabels in this case, set the axis label to a whitespce
    # beforehand, e.g.: ax.set_xlabel(' ')
    # ATTENTION: this should probably stay towards the end of this function as resizing axes/fonts can change labels)
    if b_make_ticklabel_font_sansserif:
        for ax in handle_ax:
            if any(ax.get_xlabel()):
                xticks = ax.get_xticks()
                if all([np.equal(np.mod(tick, 1), 0) for tick in xticks]):
                    ax.set_xticklabels([str(int(t)).replace('-', u"\u2212") for t in xticks],
                                       fontdict={'family': 'sans-serif'})
                else:
                    ax.set_xticklabels([str(round(t, 8)).replace('-', u"\u2212") for t in xticks],
                                       fontdict={'family': 'sans-serif'})
            if any(ax.get_ylabel()):
                yticks = ax.get_yticks()
                if all([np.equal(np.mod(tick, 1), 0) for tick in yticks]):
                    ax.set_yticklabels([str(int(t)).replace('-', u"\u2212") for t in yticks],
                                       fontdict={'family': 'sans-serif'})
                else:
                    ax.set_yticklabels([str(round(t, 8)).replace('-', u"\u2212") for t in yticks],
                                       fontdict={'family': 'sans-serif'})


def plot_traces_spikes_mod(statemons, spikemons, info, neuron_idx=0, artificial_spike_height=0,
                           b_spike_threshold=True, b_share_y_axis=True, t_const_event_marker=None,
                           t_const_span_marker=None, b_black_background=False, b_circuit_weight=True,
                           b_circuit_delay=True, b_circuit_probability=True, b_circuit_idx=False,
                           b_maxmin_marker=False, b_derivative=False):
    """Plot voltage trace of membrane potential for one model neuron per population in one subplot each, as well as
    one subplot containing spike times from all neurons. An additional subplot contains a network diagram. One figure
    is created for each brian2 StateMonitor object in argument statemons.

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param neuron_idx: [default=0] index of the neuron(s) for which the data should be plotted. Can be single integer
        or list of integers with length==number of populations. If a single value is passed and the data contain several
        populations (as per info), then the single value will be used for all populations (e.g. 0 means the first neuron
        of each population will be plotted).
    :type neuron_idx: int or list
    :param artificial_spike_height: [default=0] height in mV of vertical lines representing spikes. 0 means no lines.
    :type artificial_spike_height: int or float
    :param b_spike_threshold: [default=True] if true plots the spike threshold (value from info) as gray dotted line
    :type b_spike_threshold: bool
    :param b_share_y_axis: [default=True] if true, all voltage trace subplots share the same y-axis limits
    :type b_share_y_axis: bool
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_circuit_weight: [default=True] plot synaptic weight values next to connecting lines in circuit diagram
    :type b_circuit_weight: bool
    :param b_circuit_delay: [default=True] plot synaptic delay values next to connecting lines
    :type b_circuit_delay: bool
    :param b_circuit_probability: [default=True] plot synaptic connection probability values next to connecting lines
    :type b_circuit_probability: bool
    :param b_circuit_idx: [default=False] plot synapse index next to connecting lines
    :type b_circuit_idx: bool
    :param b_maxmin_marker: [default=False] mark maximum and minimum of each trace (if above/below initial voltage)
    :type b_maxmin_marker: bool
    :param b_derivative: plot derivative of each trace, rescaled and shifted to trace plot
    :type b_derivative: bool
    :return:
        - fig: figure handle(s)
        - axes: axis handles, one sublist per figure
    :rtype:
        - fig: [matplotlib.figure.Figure]
        - axes: [[matplotlib.axes._subplots.AxesSubplot]]
    """

    # if statemons and spikemons contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(info) is not list:
        info = [info]
    if type(neuron_idx) is not list:
        neuron_idx = [neuron_idx]

    # check inputs
    assert all(isinstance(item, b2.monitors.statemonitor.StateMonitor) for item in statemons) or \
        all(isinstance(item, SimpleNamespace) for item in statemons), \
        "statemons must be a list of b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert all(isinstance(item, int) for item in neuron_idx), \
        "neuron_idx must be an integer or list of integers"
    assert len(statemons) is len(spikemons) is len(info), \
        "statemons, spikemons and info must all have the same number of elements"
    assert isinstance(artificial_spike_height, int) or \
           isinstance(artificial_spike_height, float), "artificial_spike_height must be int or float"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')
    fig = []
    axes = []
    axes_current = []
    n_figs = len(statemons)
    n_gridcols = 5
    b_input_current_has_been_plotted = False

    if n_figs > 20:
        print("WARNING: More than 20 figures are about to be created...")

    # loop through list of state monitors
    for mon in range(n_figs):
        # if single neuron index was supplied, repeat that for each population
        n_populations = info[mon]['n_populations']
        if len(neuron_idx) == 1 and n_populations > 1:
            nrn_idx_per_pop = neuron_idx * n_populations
        elif len(neuron_idx) == n_populations:
            nrn_idx_per_pop = neuron_idx
        else:
            raise ValueError("Argument neuron_idx must be a single integer or a list of integers with"
                             " length==number of populations")
        # test if any neuron index is outside of the range of neurons for any of the populations
        if not all([nrn_idx_per_pop[i] < info[mon]['population_sizes'][i] for i in range(n_populations)]):
            raise ValueError("Neuron indices " + str(nrn_idx_per_pop) + " are outside of the range of neurons in the"
                             " populations of sizes " + str(info[mon]['population_sizes']))
        # get absolute neuron indices, i.e. the index to the totality of neurons, independent of the neuron's population
        nrn_idx_abs = [nrn_idx_per_pop[p] + sum(info[mon]['population_sizes'][:p]) for p in range(n_populations)]
        print(". plotting neurons " + str(nrn_idx_abs) + " (neurons " + str(nrn_idx_per_pop) +
              " of their respective populations of size " + str(info[mon]['population_sizes']) + ")")

        # create figure and set up grid for subplots
        n_gridrows = n_populations + 1  # last row is for spike times
        fig_tmp = plt.figure(figsize=(16, 9))
        fig.append(fig_tmp)
        plt.rcParams.update({'font.size': 12})
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
        axes.append([])
        axes_current.append([])
        color_map_light = get_color_list(n_populations, color_map_in=C_COLORS_LIGHT)
        color_map_dark = get_color_list(n_populations, color_map_in=C_COLORS)

        # loop through populations in current statemonitor and plot the voltage trace for the example neuron in each pop
        max_voltage = None
        min_voltage = None
        for pop in range(n_populations):
            # absolute index of current neuron
            nrn = nrn_idx_abs[pop]
            # set up subplot
            axes[mon].append(plt.subplot2grid((n_gridrows, n_gridcols), (pop, 0), colspan=n_gridcols-1, rowspan=1))
            axes[mon][pop].set_ylabel('v [mV]')
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axes[mon][pop].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            if t_const_span_marker is not None:
                axes[mon][pop].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            # plot threshold potential
            v_thresh_cur_nrn = info[mon]['v_thresh'][nrn]
            if b_spike_threshold:
                axes[mon][pop].plot([0, info[mon]['sim_time']], [v_thresh_cur_nrn, v_thresh_cur_nrn],
                                    linestyle='--', color=C_COLORS_GREY[6])
                if pop == 0:
                    plt.text(info[mon]['sim_time'] / 100.0, v_thresh_cur_nrn, 'spike threshold', color=C_COLORS_GREY[6],
                             verticalalignment='top')
            # plot input current (if non-zero)
            axis_current_tmp = axes[mon][pop].twinx()
            axes_current[mon].append(axis_current_tmp)
            if hasattr(statemons[mon], 'Ie') and any(statemons[mon].Ie[nrn, :] > 0):
                axes_current[mon][pop].plot(statemons[mon].t / b2.ms, statemons[mon].Ie[nrn, :] / b2.nA,
                                            color=color_map_light[pop], linewidth=1, linestyle=':')
                axes_current[mon][pop].set_ylabel('Ie [nA]')
                b_input_current_has_been_plotted = True
            # plot voltage traces
            axes[mon][pop].plot(statemons[mon].t / b2.ms, statemons[mon].v[nrn, :] / b2.mV, color=color_map_dark[pop])
            # plot artificial spikes
            if artificial_spike_height:
                spiketimes_cur_nrn = spikemons[mon].t[spikemons[mon].i == nrn] / b2.ms
                for s in range(len(spiketimes_cur_nrn)):
                    axes[mon][pop].plot([spiketimes_cur_nrn[s], spiketimes_cur_nrn[s]],
                                        [v_thresh_cur_nrn, v_thresh_cur_nrn + artificial_spike_height],
                                        color=color_map_dark[pop])
            # get max/min for y-axis
            max_voltage_cur = np.max(statemons[mon].v[nrn, :] / b2.mV)
            min_voltage_cur = np.min(statemons[mon].v[nrn, :] / b2.mV)
            if max_voltage is None or max_voltage_cur > max_voltage:
                max_voltage = max_voltage_cur
            if min_voltage is None or min_voltage_cur < min_voltage:
                min_voltage = min_voltage_cur
            # plot marker at maximum / minimum voltage
            if b_maxmin_marker:
                initial_voltage = statemons[mon].v[nrn, 0] / b2.mV
                if max_voltage_cur > initial_voltage:
                    i_max_v = np.argmax(statemons[mon].v[nrn, :] / b2.mV)
                    t_max_v = statemons[mon].t[i_max_v] / b2.ms
                    axes[mon][pop].plot(t_max_v, max_voltage_cur, 'xk')
                    axes[mon][pop].text(t_max_v, max_voltage_cur, '  ttp: ' + str(round(t_max_v, 1)) + 'ms, amp: ' +
                                                                  str(round(max_voltage_cur - initial_voltage, 1)) +
                                                                  'mV', verticalalignment='top')
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_max_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axes[mon][pop].plot(t_rise, v_25_percent, 'xk')
                    axes[mon][pop].text(t_rise, v_25_percent, '  25% rt: ' + str(round(t_rise, 1)) + 'ms',
                                        verticalalignment='top')
                if min_voltage_cur < initial_voltage:
                    i_min_v = np.argmin(statemons[mon].v[nrn, :] / b2.mV)
                    t_min_v = statemons[mon].t[i_min_v] / b2.ms
                    axes[mon][pop].plot(t_min_v, min_voltage_cur, 'xk')
                    axes[mon][pop].text(t_min_v, min_voltage_cur, '  ttp: ' + str(round(t_min_v, 1)) + 'ms, amp: ' +
                                                                  str(round(min_voltage_cur - initial_voltage, 1)) +
                                                                  'mV', verticalalignment='bottom')
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_min_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axes[mon][pop].plot(t_rise, v_25_percent, 'xk')
                    axes[mon][pop].text(t_rise, v_25_percent, '  25% rt: ' + str(round(t_rise, 1)) + 'ms',
                                        verticalalignment='top')
            # plot derivative of trace
            if b_derivative:
                gradient = np.gradient(statemons[mon].v[nrn, :] / b2.mV)
                min_g = min(gradient)
                max_g = max(gradient)
                scaling_factor_g = 1 / (max_g - min_g) * (max_voltage_cur - min_voltage_cur)
                gradient_rescaled = gradient * scaling_factor_g
                gradient_shifted = gradient_rescaled - min_g * scaling_factor_g + min_voltage_cur
                axes[mon][pop].plot(statemons[mon].t / b2.ms, gradient_shifted, color=[.4, .4, .4])
                if b_maxmin_marker:
                    if abs(max_g) > abs(min_g):
                        t_max_g = statemons[mon].t[np.argmax(gradient)] / b2.ms
                        axes[mon][pop].plot(t_max_g, max(gradient_shifted), 'xk')
                        axes[mon][pop].text(t_max_g, max(gradient_shifted),
                                            '  gradient: ' + str(round(t_max_g, 2)) + 'ms', verticalalignment='bottom')
                    if abs(max_g) < abs(min_g):
                        t_min_g = statemons[mon].t[np.argmin(gradient)] / b2.ms
                        axes[mon][pop].plot(t_min_g, min(gradient_shifted), 'xk')
                        axes[mon][pop].text(t_min_g, min(gradient_shifted),
                                            '  gradient: ' + str(round(t_min_g, 2)) + 'ms', verticalalignment='top')

        # set up subplot in last grid row for spike times
        axes[mon].append(plt.subplot2grid((n_gridrows, n_gridcols), (n_gridrows-1, 0), colspan=n_gridcols-1, rowspan=1))
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
        if t_const_event_marker is not None:
            axes[mon][n_gridrows-1].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
        if t_const_span_marker is not None:
            axes[mon][n_gridrows-1].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
        # plot spike times
        for pop in range(n_populations):
            # absolute index of current neuron
            nrn = nrn_idx_abs[pop]
            # plot spike events in last subplot
            axes[mon][n_gridrows-1].eventplot(spikemons[mon].t[spikemons[mon].i == nrn] / b2.ms,
                                              color=color_map_dark[pop], lineoffsets=pop)
        # add right hand axis for label "spike times"
        axis_right = axes[mon][n_gridrows-1].twinx()
        axis_right.set_ylabel('Spike times')
        axis_right.tick_params(which='both', right=False, labelright=False)

        # share y axis among voltage trace plots and x axis (time) among all horizontal plots
        [axes[mon][n].get_shared_x_axes().join(axes[mon][n], axes[mon][n+1]) for n in range(n_gridrows-1)]
        if b_share_y_axis:
            [axes[mon][n].get_shared_y_axes().join(axes[mon][n], axes[mon][n+1]) for n in range(n_gridrows-2)]
            [axes_current[mon][n].get_shared_y_axes().join(axes_current[mon][n],
                                                           axes_current[mon][n+1]) for n in range(n_gridrows-2)]
        if not b_input_current_has_been_plotted:
            [axes_current[mon][n].set_yticks([]) for n in range(n_gridrows-1)]
        # disable x tick labels for voltage trace plots and autoscale all axes
        [axes[mon][n].set_xticklabels([]) for n in range(n_gridrows-1)]
        [axes[mon][n].autoscale() for n in range(n_gridrows-1)]
        [axes_current[mon][n].autoscale() for n in range(n_gridrows-2)]

        # set axis labels
        if b_share_y_axis:
            [axes[mon][n].set_ylim(min_voltage - 5, max_voltage + artificial_spike_height + 5)
                for n in range(n_gridrows - 2)]
            [axes_current[mon][n].set_ylim(0, 1) for n in range(n_gridrows - 2)]

        # set y tick labels of spike plot to population ids, and other settings
        axes[mon][n_gridrows-1].set_yticks(range(n_gridrows - 1))
        nrn_ids_text = [str(nrn_idx_per_pop[pop] + 1) + '/' + str(info[mon]['population_sizes'][pop])
                        for pop in range(n_populations)]
        axes[mon][n_gridrows-1].set_yticklabels(nrn_ids_text)
        axes[mon][n_gridrows-1].set_ylabel('Neuron #')
        axes[mon][n_gridrows-1].set_ylim(n_gridrows-1.5, -0.5)
        axes[mon][0].set_xlim(0, info[mon]['sim_time'])
        axes[mon][n_gridrows-1].set_xlabel('t [ms]')

        # plot network diagram in right grid column
        axis_net = plt.subplot2grid((n_gridrows, n_gridcols), (0, n_gridcols-1), rowspan=n_gridrows - 1, colspan=1)
        axes[mon].append(axis_net)
        plot_circuit_diagram(axis_net, info[mon], b_weight=b_circuit_weight, b_delay=b_circuit_delay,
                             b_probability=b_circuit_probability, b_syn_idx=b_circuit_idx)

    return fig, axes


def plot_traces_mod(statemons, info, b_spike_threshold=True, b_average_aggregate=False,
                    t_const_event_marker=None, t_const_span_marker=None, marker_linestyle='--', t_offset_ms=0,
                    b_black_background=False, trace_color=None, average_color=None, h_ax=None):
    """Plot voltage traces of ALL model neurons of one run into one figure, one subplot per neuron.

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param b_spike_threshold: [default=True] if true plots the spike threshold (value from info) as gray dotted line
    :type b_spike_threshold: bool
    :param b_average_aggregate: if true, treat statemons as aggregate of parex and plot all traces in single figure
    :type b_average_aggregate: bool
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param marker_linestyle: linestyle for event marker (default: '--' = dashed line)
    :type marker_linestyle: str
    :param t_offset_ms: [default=0] time value in milliseconds to offset the plot by (e.g. to align to an event)
    :type t_offset_ms: int or float
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param trace_color: single color value to use for all traces
    :type trace_color: list or None
    :param average_color: single color value to use for trace average
    :type average_color: list or None
    :param h_ax: handle to the axis in which the traces are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle(s)
        - axes: axis handles, one sublist per figure
    :rtype:
        - fig: [matplotlib.figure.Figure]
        - axes: [[matplotlib.axes._subplots.AxesSubplot]]
    """

    # if statemons and info contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(statemons) is len(info), \
        "statemons and info must have the same number of elements"
    if h_ax and not b_average_aggregate:
        print("WARNING: Plotting to a specified axis (arg h_ax) might only work with b_average_aggregate")

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')

    # prepare figure and axis
    fig = []
    axes = []
    n_figs = len(statemons)
    if n_figs > 20 and not b_average_aggregate:
        print("WARNING: More than 20 figures are about to be created...")

    # get absolute minimum, maximum and range of all traces for y-axis limits
    abs_min_trc = np.min([np.min(statemons[m].v / b2.mV) for m in range(len(statemons))])
    abs_max_trc = np.max([np.max(statemons[m].v / b2.mV) for m in range(len(statemons))])

    # loop through list of state monitors
    idx_ax = -1  # counter for axis list (only gets incremented when plotting multiple figures
    idx_ax_nrn = -1  # counter for axis list (only gets incremented when plotting multiple figures
    for mon in range(len(statemons)):
        # create figure and set up grid for subplots (if b_average_aggregate, only create one figure
        n_neurons = sum(info[mon]['population_sizes'])
        if not b_average_aggregate or (b_average_aggregate and mon == 0):
            if not h_ax:
                fig_tmp = plt.figure(figsize=(16, 9))
                fig_tmp.tight_layout()
                fig_tmp.suptitle(info[mon]['filename'] + ' - populations: ' + str(info[mon]['population_ids']))
                fig.append(fig_tmp)
                plt.rcParams.update({'font.size': 9})
            axes.append([])
            idx_ax += 1
        # set up subplots
        if n_neurons > 144:
            n_gridcols = 12
        else:
            n_gridcols = int(np.ceil(n_neurons/12))
        n_gridrows = np.ceil(n_neurons / n_gridcols).astype(int)
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
        n_populations = info[mon]['n_populations']
        color_map_dark = get_color_list(n_populations, color_map_in=C_COLORS)

        if not b_average_aggregate:
            idx_ax_nrn = -1
        for nrn in range(n_neurons):
            # get population index of current neuron
            idx_nrn_relative, idx_pop_cur_nrn = p_util.get_rel_from_abs_nrn_idx(nrn, info[mon]['population_sizes'])
            # set up subplot
            [sp_col, sp_row] = np.divmod(nrn, n_gridrows)
            if not b_average_aggregate or (b_average_aggregate and nrn == 0 and mon == 0):
                if not h_ax:
                    axes[idx_ax].append(plt.subplot2grid((n_gridrows, n_gridcols), (sp_row, sp_col),
                                                         colspan=1, rowspan=1))
                else:
                    axes[idx_ax].append(h_ax)
                idx_ax_nrn += 1
                # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that time
                if t_const_event_marker is not None:
                    axes[idx_ax][idx_ax_nrn].axvline(t_const_event_marker, color='k', linewidth=1,
                                                     linestyle=marker_linestyle, zorder=2.5)
                if t_const_span_marker is not None:  # for "playback" text, see plot_traces_spikes_mod()
                    axes[idx_ax][idx_ax_nrn].axvspan(t_const_span_marker[0], t_const_span_marker[1],
                                                     facecolor=C_COLORS_GREY[9])
                # plot threshold potential
                if b_spike_threshold:
                    v_thresh_cur_nrn = info[mon]['v_thresh'][nrn]
                    axes[idx_ax][idx_ax_nrn].plot([0, info[mon]['sim_time']], [v_thresh_cur_nrn, v_thresh_cur_nrn],
                                                  linestyle='--', color=C_COLORS_GREY[6])
                    if nrn == 0:
                        plt.text(info[mon]['sim_time'] / 100.0, v_thresh_cur_nrn, 'spike threshold',
                                 color=C_COLORS_GREY[6], verticalalignment='top')
            # plot voltage traces
            if trace_color is None:
                color_cur = color_map_dark[idx_pop_cur_nrn]
            else:
                color_cur = trace_color
            if b_average_aggregate and mon == 0:
                label = 'model'
            else:
                label=''
            axes[idx_ax][idx_ax_nrn].plot(statemons[mon].t / b2.ms - t_offset_ms, statemons[mon].v[nrn, :] / b2.mV,
                                          color=color_cur, linewidth=1, label=label)
            # plot neuron index
            if not b_average_aggregate:
                axes[idx_ax][idx_ax_nrn].text(info[mon]['sim_time'], abs_min_trc,
                                              str(nrn) + '|' + str(idx_nrn_relative), color='gray',
                                              verticalalignment='bottom', horizontalalignment='right', fontsize=5)
            # to plot input current, see plot_traces_spikes_mod()
            axes[idx_ax][idx_ax_nrn].set_ylim(abs_min_trc, abs_max_trc)
            if nrn < n_neurons-1:
                axes[idx_ax][idx_ax_nrn].axes.xaxis.set_ticklabels([])
            else:
                axes[idx_ax][idx_ax_nrn].set_xlabel('t [ms]')
            if nrn > 0:
                axes[idx_ax][idx_ax_nrn].axes.yaxis.set_ticklabels([])

        # share x axis (time) among all  plots
        if n_neurons > 1:
            [axes[idx_ax][n].get_shared_x_axes().join(axes[idx_ax][n], axes[idx_ax][n+1])
                for n in range(n_neurons-1)]

        # other settings
        axes[idx_ax][0].set_ylabel('v [mV]')
        axes[idx_ax][0].set_xlim(0, info[mon]['sim_time'])

    # plot mean if aggregate
    if b_average_aggregate:
        if average_color is None:
            average_color = 'k'
        axes[idx_ax][idx_ax_nrn].plot(statemons[0].t / b2.ms - t_offset_ms,
                                      np.mean([s.v / b2.mV for s in statemons], axis=0)[0], color=average_color,
                                      linewidth=2, label='average', zorder=3)
        if not h_ax:
            fig_tmp.suptitle(info[0]['filename'] + ' run ' + str(info[0]['run_id']).zfill(4) +
                             ' - all traces from aggregate plus average (black)')
        axes[idx_ax][idx_ax_nrn].legend(prop={'size': 8})

    if not h_ax:
        return fig, axes
    else:
        return


def plot_traces_rec(traces, spiketimes_ms_in=None, sampling_frequency_khz=40, b_detect_spikes=False,
                    b_detect_onsets=False, b_subplots=False, b_plot_raw_traces=True, b_plot_smoothed=False,
                    b_plot_derivative=False, b_plot_average=False, b_n_in_legend=True, b_offset_by_average=False,
                    b_average_mean_2std=False, b_std_interval=False, t_baseline_ms=None, t_const_event_marker=None,
                    t_const_span_marker=None, marker_linestyle='--', trace_color_smooth='xkcd:orangered',
                    average_color='b', t_offset_ms=0, h_ax=None):
    """Plot one or more recorded voltage traces in a single figure.

    :param traces: (list of) numpy ndarray(s) of voltage traces
    :type traces: np.ndarray or list
    :param spiketimes_ms_in: [default=None] optional list of spiketimes (see return). If None and b_detect_spikes=True,
        spiketimes will be calculated
    :type spiketimes_ms_in: [[float]] or None
    :param sampling_frequency_khz: [default=40] sampling frequency of traces in kHz
    :type sampling_frequency_khz: int or float
    :param b_detect_spikes: [default=False] detect spikes and mark spike peaks in plot
    :type b_detect_spikes: bool
    :param b_detect_onsets: [default=False] detect and mark spike onsets
    :type b_detect_onsets: bool
    :param b_subplots: [default=False] if multiple traces were passed, plot them in separate subplots, not overlaid
    :type b_subplots: bool
    :param b_plot_raw_traces: [default=True] plot individual raw traces (turn of if e.g. only plotting average)
    :type b_plot_raw_traces: bool
    :param b_plot_smoothed: [default=False] plot the smoothed traces (required to plot derivative)
    :type b_plot_smoothed: bool
    :param b_plot_derivative: [default=False] plot the derivative of each trace (depends on b_plot_smoothed)
    :type b_plot_derivative: bool
    :param b_plot_average: [default=False] plot average of traces (if b_subplots is false)
    :type b_plot_average: bool
    :param b_n_in_legend: [default=True] if False, don't plot number of traces in legend
    :type b_n_in_legend: bool
    :param b_offset_by_average: [default=False] offset each individual trace by its average (e.g. to correct for drift)
    :type b_offset_by_average: bool
    :param b_average_mean_2std: [default=False] plot horizontal lines for mean and 2 * standard deviation of average
        within t_baseline
    :type b_average_mean_2std: bool
    :param b_std_interval: [default=False] plot interval +/- 1 standard deviation around average
    :type b_std_interval: bool
    :param t_baseline_ms: [default=None] two-element list of start and end in milliseconds of baseline for mean & std
    :type t_baseline_ms: list or None
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param marker_linestyle: linestyle for event marker (default: '--' = dashed line)
    :type marker_linestyle: str
    :param trace_color_smooth: single color value to use for all smoothed traces
    :type trace_color_smooth: list or str or None
    :param average_color: single color value to use for trace average
    :type average_color: list or str or None
    :param t_offset_ms: [default=0] time value in milliseconds to offset the plot by (e.g. to align to an event)
    :type t_offset_ms: int or float
    :param h_ax: handle to the axis in which the traces are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values. Only works if b_subplots==False.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - axes: list of axis handles
        - spiketimes_ms_calculated: list of list(s) of spike times (in milliseconds). One sublist per trace.
    :rtype:
        - fig: matplotlib.figure.Figure
        - axes: [matplotlib.axes._subplots.AxesSubplot]
        - spiketimes_ms_calculated: [[float]]
    """

    # if traces contains a single trace, convert it to a list
    if type(traces) is not list:
        traces = [traces]
    if type(sampling_frequency_khz) is not list:
        sampling_frequency_khz = [sampling_frequency_khz]
    if len(sampling_frequency_khz) == 1 and len(traces) > 1:
        sampling_frequency_khz = sampling_frequency_khz * len(traces)

    # check inputs
    assert all(isinstance(item, np.ndarray) for item in traces), \
        "traces must be a numpy ndarray or list of ndarrays"
    if b_plot_derivative and not b_plot_smoothed:
        b_plot_derivative = False
        print("WARNING: gradient not plotted, because b_plot_smoothed == False")
    if b_detect_onsets:
        assert spiketimes_ms_in or b_detect_spikes, "spike onset detection only possible if " + \
            "spiketimes_ms_in is not None or b_detect_spikes is True"
    if h_ax:
        assert not b_subplots, "plotting in passed axis only possible if b_subplots == False"

    # create figure
    if not h_ax:
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({'font.size': 15})

    # misc variables
    n_traces = len(traces)
    max_dur = 0
    smoothing_filter_window_length = 11

    # offset traces by their average (e.g. to correct for long term drift during recording)
    if b_offset_by_average:
        for trc in range(n_traces):
            traces[trc] = traces[trc] - np.mean(traces[trc])

    # smooth traces using savitzky-golay filter (from scipy.signal)
    if b_plot_smoothed:
        traces_smooth_sg = []
        for trc in range(n_traces):
            traces_smooth_sg.append(savgol_filter(traces[trc], smoothing_filter_window_length, 3))

    # get absolute minimum, maximum and range of all traces for y-axis limits
    abs_min_trc = np.min([np.min(traces[t]) for t in range(len(traces))])
    abs_max_trc = np.max([np.max(traces[t]) for t in range(len(traces))])
    abs_rng_trc = abs_max_trc - abs_min_trc

    # calculate the derivative / gradient of each trace and set negatives value to zero
    if b_plot_derivative:
        gradients = []
        for trc in range(n_traces):
            gradients.append(np.gradient(traces_smooth_sg[trc]))
            gradients[trc][gradients[trc] < 0] = 0
        # get absolute minimum, maximum and range of the derivative / gradient of each trace
        abs_min_grd = np.min([np.min(gradients[t]) for t in range(len(gradients))])
        abs_max_grd = np.max([np.max(gradients[t]) for t in range(len(gradients))])
        abs_rng_grd = abs_max_grd - abs_min_grd
        # normalize derivatives / gradients to the absolute range of the traces
        for trc in range(n_traces):
            gradients[trc] = (gradients[trc] - abs_min_grd) * (abs_rng_trc / abs_rng_grd) + abs_min_trc

    # set up subplots
    axes = []
    if b_subplots:
        if n_traces < 9:
            n_gridcols = 1
        elif n_traces < 25:
            n_gridcols = 2
        elif n_traces < 40:
            n_gridcols = 3
        elif n_traces < 80:
            n_gridcols = 4
        else:
            n_gridcols = 5
        n_gridrows = np.ceil(n_traces / n_gridcols).astype(int).item()
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    else:
        if not h_ax:
            axes.append(plt.axes())
        else:
            axes.append(h_ax)
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
        if t_const_event_marker is not None:
            axes[0].axvline(t_const_event_marker, color='k', linewidth=1, linestyle=marker_linestyle, zorder=2.5)
        if t_const_span_marker is not None:
            axes[0].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])

    # detect spiketimes (optional)
    if b_detect_spikes:
        [spiketimes_grd, _, thresholds_used_grd] = \
            p_analysis.get_spiketimes(traces, smooth_filter_win_len=smoothing_filter_window_length)
        spiketimes_ms_calculated = []
    else:
        spiketimes_grd = []
        spiketimes_ms_calculated = []

    # loop through traces and plot
    for trc in range(n_traces):
        n_values = traces[trc].size
        dur_sample = 1 / sampling_frequency_khz[trc]
        dur_trace = dur_sample * n_values
        x_values = np.linspace(0, dur_trace, n_values)
        # set up axis (if one subplot per trace)
        if b_subplots:
            [sp_col, sp_row] = np.divmod(trc, n_gridrows)
            axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (sp_row, sp_col), colspan=1, rowspan=1)
            axes.append(axis_tmp)
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axis_tmp.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1, linestyle=marker_linestyle)
            if t_const_span_marker is not None:
                axis_tmp.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            # plot spike detection threshold (and normalize to the absolute range used to plot traces)
            if b_detect_spikes and b_plot_derivative:
                thresh_for_plot = (thresholds_used_grd[trc] - abs_min_grd) * (abs_rng_trc / abs_rng_grd) + abs_min_trc
                axis_tmp.plot([0, dur_trace], [thresh_for_plot, thresh_for_plot], linestyle='--',
                              color='xkcd:lightblue', label='gradient theshold')

        # plot derivatives / gradients
        if b_plot_derivative:
            if not b_subplots and trc > 0:
                plt.plot(x_values - t_offset_ms, gradients[trc], color='xkcd:azure', linewidth=1)
            else:
                plt.plot(x_values - t_offset_ms, gradients[trc], color='xkcd:azure', linewidth=1, label='gradient')

        # plot raw traces
        if b_plot_raw_traces:
            if not b_subplots and trc > 0:
                plt.plot(x_values - t_offset_ms, traces[trc], 'k')
            else:
                plt.plot(x_values - t_offset_ms, traces[trc], 'k', label='recorded trace [mV]')

        # plot smoothed trace
        if b_plot_smoothed:
            if not b_subplots and trc > 0:
                plt.plot(x_values - t_offset_ms, traces_smooth_sg[trc], color=trace_color_smooth)
            else:
                plt.plot(x_values - t_offset_ms, traces_smooth_sg[trc], color=trace_color_smooth,
                         label='smoothed recorded trace [mV]')

        # mark calculated spike peaks
        if b_detect_spikes:
            if len(spiketimes_grd) >= trc-1:
                # plt.plot([t / sampling_frequency_hz for t in spiketimes[trc]], traces[trc][spiketimes[trc]], '+k',
                #          markersize=14)

                if not b_subplots and trc > 0:
                    plt.plot(np.array([x_values[t] for t in spiketimes_grd[trc]]) - t_offset_ms,
                             traces[trc][spiketimes_grd[trc]],
                             'o', color='xkcd:orangered', markersize=10, fillstyle='none')
                else:
                    plt.plot(np.array([x_values[t] for t in spiketimes_grd[trc]]) - t_offset_ms,
                             traces[trc][spiketimes_grd[trc]],
                             'o', color='xkcd:orangered', markersize=10, fillstyle='none', label='detected spikes')
                # convert spiketimes from samples to milliseconds
                spiketimes_ms_calculated.append([spiketimes_grd[trc][idx] / sampling_frequency_khz[trc]
                                                 for idx in range(len(spiketimes_grd[trc]))])
        # if (loaded?) spiketimes were passed, also plot those
        if spiketimes_ms_in:
            spiketimes_samp_in = [np.ceil(spiketimes_ms_in[trc][s]*sampling_frequency_khz[trc])
                                  for s in range(len(spiketimes_ms_in[trc]))]
            if not b_subplots and trc > 0:
                plt.plot(spiketimes_ms_in[trc], [traces[trc][s.astype(int)] for s in spiketimes_samp_in], '+g',
                         markersize=10)
            else:
                plt.plot(spiketimes_ms_in[trc], [traces[trc][s.astype(int)] for s in spiketimes_samp_in], '+g',
                         markersize=10, label='loaded spikes')

        # detect spike onsets
        if b_detect_onsets:
            if spiketimes_ms_in:
                spike_onsets_ms, spike_thresholds = p_analysis.get_onset_of_spikes(traces[trc],
                    [spiketimes_ms_in[trc]], sampling_frequency_khz=sampling_frequency_khz[trc])
            elif b_detect_spikes:
                spike_onsets_ms, spike_thresholds = p_analysis.get_onset_of_spikes(traces[trc],
                    [spiketimes_ms_calculated[trc]], sampling_frequency_khz=sampling_frequency_khz[trc])
            plt.plot(spike_onsets_ms[0], spike_thresholds[0], 'xb', markersize=10, label='spike onsets')
            print("spike onsets trace " + str(trc) + ": " + str(spike_onsets_ms))
            print("spike thresholds trace " + str(trc) + ": " + str(spike_thresholds))

        # keep track of maximum duration for axis limits
        if dur_trace > max_dur:
            max_dur = dur_trace

        # set axis limits etc. for each subplot
        if b_subplots:
            axes[trc].set_xlim(0, max_dur)
            axes[trc].set_ylim(abs_min_trc, abs_max_trc)
            if trc < n_traces-1:
                axes[trc].axes.xaxis.set_ticklabels([])
                axes[trc].axes.yaxis.set_ticklabels([])
            else:
                axes[trc].set_xlabel('Time [s]')
                axes[trc].legend(prop={'size': 8})

    # plot average of all traces (or smoothed traces if only those are plotted)
    if not b_subplots and b_plot_average:
        if b_plot_smoothed:
            average_trace = np.mean(traces_smooth_sg, 0)
            average_label = 'smoothed recorded trace average [mV]'
        else:
            average_trace = np.mean(traces, 0)
            average_label = 'recorded trace average [mV]'
        if b_average_mean_2std and t_baseline_ms:
            average_mean = np.mean(average_trace[round(t_baseline_ms[0] * sampling_frequency_khz[0]):
                                                 round(t_baseline_ms[1] * sampling_frequency_khz[0])])
            average_std = np.std(average_trace[round(t_baseline_ms[0] * sampling_frequency_khz[0]):
                                               round(t_baseline_ms[1] * sampling_frequency_khz[0])])
            plt.axhline(average_mean, color=[.6, .6, .6], linewidth=1, linestyle='-')
            plt.axhline(average_mean + 2 * average_std, color=[.6, .6, .6], linewidth=1, linestyle='--')
            plt.axhline(average_mean - 2 * average_std, color=[.6, .6, .6], linewidth=1, linestyle='--')
        if b_std_interval:
            std_trace = np.std(traces_smooth_sg, 0)
            avg_plus_std_trc = average_trace + std_trace
            avg_minus_std_trc = average_trace - std_trace
            plt.plot(x_values - t_offset_ms, avg_plus_std_trc, '--', color=average_color, zorder=2.9, label='+/- 1 std')
            plt.plot(x_values - t_offset_ms, avg_minus_std_trc, '--', color=average_color, zorder=2.9)
        # plot trace average
        plt.plot(x_values - t_offset_ms, average_trace, color=average_color, zorder=3, label=average_label)

    # set axis limits etc. for single plot, add extra label for legend displaying the number of traces
    if not b_subplots:
        axes[0].set_xlim(0, max_dur)
        axes[0].set_xlabel('Time [ms]')
        axes[0].set_ylabel('Potential [mV]')
        if b_n_in_legend:
            axes[0].plot(np.NaN, np.NaN, '-', color='none', label='n = ' + str(n_traces))
        axes[0].legend(prop={'size': 9})

    if not h_ax:
        return fig, axes, spiketimes_ms_calculated
    else:
        return


def plot_isi_histogram(spiketimes_ms_in, b_subplots=False, title='', alpha=1.0, axis_handle=None, label='',
                       t_const_event_marker=None):
    """Plot the inter-spike interval (ISI) histograms of one or more neurons/traces in a single figure.

    :param spiketimes_ms_in: list of list(s) of spiketimes (in milliseconds). One sublist per trace. If None,
        spiketimes will be calculated
    :type spiketimes_ms_in: [[float]] or None
    :param b_subplots: [default=False] if multiple traces were passed, plot the histograms in separate subplots,
        rather than one overall histogram of all ISIs
    :type b_subplots: bool
    :param title: figure title to be plotted above histogram
    :type title: str
    :param alpha: alpha-level of the histogram bars. set to e.g. 0.5 if plotting multiple histograms in same axis
    :type alpha: float
    :param axis_handle: [default=None] handle of axis to plot in. if None, create a new figure.
    :type axis_handle: matplotlib.axes._subplots.AxesSubplot
    :param label: [default=''] label for histogram (useful if plotting multiple histograms into same axis).
        note: legend has to be added to axis after function call!
    :type label: str
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker for a event
    :type t_const_event_marker: int or float or None
    :return:
        - fig: figure handle (if one was created in this function)
        - axes: list of axis handles
    :rtype:
        - fig: matplotlib.figure.Figure or None
        - axes: [matplotlib.axes._subplots.AxesSubplot]
    """

    # check inputs
    assert isinstance(spiketimes_ms_in, list), "spiketimes_ms_in must be a list of list(s) of spiketimes"
    assert all(isinstance(item, list) for item in spiketimes_ms_in), \
        "spiketimes_ms_in must be a list of list(s) of spiketimes"
    assert isinstance(alpha, float) and 0.0 <= alpha <= 1.0, "alpha must be a float between 0.0 and 1.0"
    if axis_handle:
        assert b_subplots is False, "if axis handle is supplied, b_subplots must be false"

    # misc variables
    n_traces = len(spiketimes_ms_in)
    n_spikes_all = len(np.concatenate(spiketimes_ms_in))
    label = label + ' (' + str(n_spikes_all) + ')'
    max_dur = 0

    # create figure
    if not axis_handle:
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle(title)
        plt.rcParams.update({'font.size': 15})
    else:
        fig = None

    # set up subplots
    if b_subplots:
        axes = []
        if n_traces < 9:
            n_gridcols = 1
        elif n_traces < 25:
            n_gridcols = 2
        elif n_traces < 37:
            n_gridcols = 4
        elif n_traces < 49:
            n_gridcols = 5
        else:
            n_gridcols = 6
        n_gridrows = np.ceil(n_traces / n_gridcols).astype(int)
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    else:
        # if no axis handle was passed, create a new one
        if not axis_handle:
            axis_handle = plt.axes()
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
        if t_const_event_marker is not None:
            axis_handle.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)

    # loop through traces (i.e. sublists of spiketimes_ms_in), calculate ISIs and plot histogram
    isis = []
    for trc in range(n_traces):
        # calculate ISIs
        n_spikes = len(spiketimes_ms_in[trc])
        isis.append([spiketimes_ms_in[trc][i + 1] - spiketimes_ms_in[trc][i] for i in range(n_spikes - 1)])
        # set up axis (if one subplot per trace)
        if b_subplots:
            [sp_col, sp_row] = np.divmod(trc, n_gridrows)
            axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (sp_row, sp_col), colspan=1, rowspan=1)
            axes.append(axis_tmp)
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axis_tmp.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            # plot histogram of ISI
            if isis[trc]:
                axes[trc].hist(isis[trc], bins=np.ceil(np.max(isis[trc]) / 2).astype(int),
                               alpha=alpha, label=label, density=True)
                if np.max(isis[trc]) > max_dur:
                    max_dur = np.max(isis[trc])
            # set ticks and labels
            if trc == n_traces-1:
                axes[trc].set_xlabel('ISI [ms]')
            else:
                axes[trc].axes.xaxis.set_ticklabels([])
            if trc == 0:
                axes[trc].set_ylabel('Rel. frequency')
            else:
                axes[trc].axes.yaxis.set_ticklabels([])
            # twin axis and add number of spikes as right-hand axis label
            axis_right = axes[trc].twinx()
            axis_right.set_ylabel(str(n_spikes))
            axis_right.tick_params(which='both', right=False, labelright=False)

    # set axis limits etc. for single plot
    if not b_subplots:
        # plot histogram of ISI
        if any([isis[v] for v in range(len(isis))]):
            axis_handle.hist(np.concatenate(isis), bins=np.ceil(np.max(np.concatenate(isis))).astype(int), alpha=alpha,
                             label=label, density=True)
        axis_handle.set_xlabel('ISI [ms]')
        axis_handle.set_ylabel('Rel. frequency')
        axes = [axis_handle]
    else:
        # set x-axis limits for all
        for ax in range(len(axes)):
            axes[ax].set_xlim(0, max_dur)

    # output shortest isi
    if any([isis[v] for v in range(len(isis))]):
        print(title + " shortest ISI: " + str(np.min(np.concatenate(isis))))

    return fig, axes


def plot_traces_per_run(statemons, config, info, neuron_idx=0, population_idx=None, b_spike_threshold=True,
                        virtual_threshold=None, t_const_event_marker=None, t_const_span_marker=None, t_offset_ms=0,
                        b_black_background=False, b_maxmin_marker=False, find_max_min_from_ms=0,
                        b_legend=True, b_clipon=True, linewidth_traces=1.5, colormap=CMAP_NAME_CONTINUOUS, h_ax=None):
    """Plot voltage trace of membrane potential for multiple runs with a varying (free) parameter in one figure.

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param neuron_idx: [default=0] index of the neuron(s) for which the data should be plotted. Can be single integer
        or list of integers with length==number of populations. If a single value is passed and the data contain several
        populations (as per info), then the single value will be used for all populations (e.g. 0 means the first neuron
        of each population will be plotted).
    :type neuron_idx: int or list
    :param population_idx: [default=None] index or list of indices of the population(s) for which traces are plotted.
        If None, plot the nth nrn of all pops.
    :type population_idx: int or list or None
    :param b_spike_threshold: [default=True] if true plots the spike threshold (value from info) as gray dotted line
    :type b_spike_threshold: bool
    :param virtual_threshold: [default=None] if not None, plot spike threshold here [mV] instead of the actual value
    :type virtual_threshold: float or int or None
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param t_offset_ms: [default=0] time value in milliseconds to offset the plot by (e.g. to align to an event)
    :type t_offset_ms: int or float
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_maxmin_marker: [default=False] mark maximum and minimum of each trace (if above/below initial voltage)
    :type b_maxmin_marker: bool
    :param find_max_min_from_ms: [default=0] look for maximum/minimum in trace after this time in milliseconds.
        only used if b_maxmin_marker is True. Should equal onset of presynaptic spike, as it's used in rise time calc
    :type find_max_min_from_ms: int
    :param b_legend: [default=True] add legend to plot
    :type b_legend: bool
    :param b_clipon: [default=True] if False, allows to plot beyond axis limits
    :type b_clipon: bool
    :param linewidth_traces: [default=1.5] linewidth for voltage traces
    :type linewidth_traces: float or int
    :param colormap: [default='viridis' (defined by CMAP_NAME_CONTINUOUS)] name of matplotlib color map to use for trcs
    :type colormap: str
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot

    :return:
        - fig: figure handle
        - axes: list of axis handles
    :rtype:
        - fig: matplotlib.figure.Figure
        - axes: [matplotlib.axes._subplots.AxesSubplot]
    """

    # if statemons, info and neuron_idx contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(info) is not list:
        info = [info]
    if type(neuron_idx) is not list:
        neuron_idx = [neuron_idx]
    if population_idx is not None and type(population_idx) is not list:
        population_idx = [population_idx]

    # check inputs
    assert all(isinstance(item, b2.monitors.statemonitor.StateMonitor) for item in statemons) or \
        all(isinstance(item, SimpleNamespace) for item in statemons), \
        "statemons must be a list of b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert all(isinstance(item, int) for item in neuron_idx), \
        "neuron_idx must be an integer or list of integers"
    if population_idx is not None:
        assert all(isinstance(item, int) for item in population_idx), \
            "population_idx must be an integer or list of integers (or None to plot a neuron from all populations)"
    assert len(statemons) is len(info), \
        "statemons and info must have the same number of elements"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')

    # get populations to plot
    n_populations_total = info[0]['n_populations']
    if population_idx:
        n_populations = len(population_idx)
    else:
        n_populations = n_populations_total
        population_idx = list(range(n_populations))

    # if single neuron index was supplied, repeat that for each population
    if len(neuron_idx) == 1 and n_populations > 1:
        nrn_idx_per_pop = neuron_idx * n_populations
    elif len(neuron_idx) == n_populations:
        nrn_idx_per_pop = neuron_idx
    else:
        raise ValueError("Argument neuron_idx must be a single integer or a list of integers with"
                         " length==number of populations")

    # test if any neuron index is outside of the range of neurons for any of the populations
    if not all([nrn_idx_per_pop[i] < info[0]['population_sizes'][population_idx[i]] for i in range(n_populations)]):
        raise ValueError("Neuron indices " + str(nrn_idx_per_pop) + " are outside of the range of neurons in the" +
                         " population(s) " + str(population_idx))
    # get absolute neuron indices, i.e. the index to the totality of neurons, independent of the neuron's population
    nrn_idx_abs = [nrn_idx_per_pop[p] + sum(info[0]['population_sizes'][:population_idx[p]])
                   for p in range(n_populations)]
    print(". plotting (plot_traces_per_run) neurons " + str(nrn_idx_abs) + " (neurons " + str(nrn_idx_per_pop) +
          " of their respective populations" + str(population_idx) + ")")
    # if the free_parameter is a neuron parameter, subplot legend labels correspond to the free param value of the
    # respective population. if it's a synapse param, the label only takes the first free param value for all subplots
    if info[0]['free_parameter_dict'] is not 'parameters_nrn':
        print("- free parameter is a synatptic parameter => ALL LEGEND ENTRIES REFER TO FIRST VALUE OF FREE PARAM")

    # create figure and axis. if an axis handle was passed, plot only the first population to that axis
    if not h_ax:
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({'font.size': 12})
        n_gridcols = 1
        n_gridrows = n_populations
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    else:
        population_idx = [population_idx[0]]
    axes = []
    n_runs = len(statemons)
    cmap = plt.cm.get_cmap(colormap)

    # loop through values of the free parameter (either populations or synapses) and create a subplot for each
    for i_ax, pop in enumerate(population_idx):
        # absolute index of current neuron
        nrn = nrn_idx_abs[i_ax]
        mon = 0
        if not h_ax:
            axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (i_ax, 0), colspan=1, rowspan=1))
        else:
            axes.append(h_ax)
        axes[i_ax].set_ylabel('v [mV]')
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
        if t_const_event_marker is not None:
            axes[i_ax].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
        if t_const_span_marker is not None:
            axes[i_ax].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            axes[i_ax].text(t_const_span_marker[0] + (t_const_span_marker[1]-t_const_span_marker[0]) / 2,
                            np.min(statemons[mon].v[nrn, :]) / b2.mV, "playback", color=C_COLORS_GREY[6],
                            zorder=1, verticalalignment='bottom', horizontalalignment='center')
        # plot threshold potential
        if b_spike_threshold:
            if virtual_threshold is not None:
                v_thresh_cur_nrn = virtual_threshold
            else:
                v_thresh_cur_nrn = info[mon]['v_thresh'][nrn]
            axes[i_ax].plot([0, info[mon]['sim_time']], [v_thresh_cur_nrn, v_thresh_cur_nrn], linestyle='--',
                            color=C_COLORS_GREY[6])
            plt.text(info[mon]['sim_time'] / 100.0, v_thresh_cur_nrn, 'spike threshold', color=C_COLORS_GREY[6],
                     verticalalignment='top')

        # loop through list of state monitors
        h_lines = []
        for mon in reversed(range(n_runs)):
            # plot voltage traces
            if info[0]['free_parameter_dict'] == 'parameters_nrn':
                free_param_val_label = 'fpv[pop] %.3f' % float(info[mon]['free_parameter_values'][pop])
            elif info[0]['free_parameter_dict'] == 'parameters_syn' and 'plot' in config and \
                    'idx_synpop_oi_for_fp' in config['plot']:
                free_param_val_label = 'fpv %.3f' % \
                                       float(info[mon]['free_parameter_values'][config['plot']['idx_synpop_oi_for_fp']])
            elif info[0]['free_parameter_dict'] == 'input_current':
                free_param_val_label = 'fpv %.3f ?' % float(info[mon]['free_parameter_values'][0][0])
            else:
                free_param_val_label = 'fpv ?'
            h_lines.append(axes[i_ax].plot(statemons[mon].t / b2.ms - t_offset_ms,
                                           np.squeeze(statemons[mon].v[nrn, :] / b2.mV), clip_on=b_clipon,
                                           color=cmap(mon / (n_runs-1)), label=free_param_val_label,
                                           linewidth=linewidth_traces)[0])
            # plot marker at maximum / minimum voltage
            if b_maxmin_marker:
                i_inital_timepoint = np.where(statemons[mon].t / b2.ms >= find_max_min_from_ms)[0][0]
                initial_voltage = statemons[mon].v[nrn, i_inital_timepoint] / b2.mV
                max_voltage_after_initial_timepoint = np.max(statemons[mon].v[nrn, i_inital_timepoint:-1] / b2.mV)
                min_voltage_after_initial_timepoint = np.min(statemons[mon].v[nrn, i_inital_timepoint:-1] / b2.mV)
                if max_voltage_after_initial_timepoint > initial_voltage + 0.04:  # 0.04: min diff to be considered peak
                    i_max_v = i_inital_timepoint + np.argmax(statemons[mon].v[nrn, i_inital_timepoint:-1] / b2.mV)
                    t_max_v = statemons[mon].t[i_max_v] / b2.ms
                    axes[i_ax].plot(t_max_v, max_voltage_after_initial_timepoint, 'xk')
                    axes[i_ax].text(t_max_v, max_voltage_after_initial_timepoint, '  ttp: ' + str(round(t_max_v, 1)) +
                                    'ms, amp: ' + str(round(max_voltage_after_initial_timepoint - initial_voltage, 1)) +
                                    'mV', verticalalignment='top')
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_max_v,
                                                                    i_inital_timepoint, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axes[i_ax].plot(t_rise + find_max_min_from_ms, v_25_percent, 'xk')
                    axes[i_ax].text(t_rise + find_max_min_from_ms, v_25_percent,
                                    '  25% rt: ' + str(round(t_rise, 1)) + 'ms', verticalalignment='top',
                                    horizontalalignment='right')
                if min_voltage_after_initial_timepoint < initial_voltage - 0.04:  # 0.04: min diff to be considered peak
                    i_min_v = i_inital_timepoint + np.argmin(statemons[mon].v[nrn, i_inital_timepoint:-1] / b2.mV)
                    t_min_v = statemons[mon].t[i_min_v] / b2.ms
                    axes[i_ax].plot(t_min_v, min_voltage_after_initial_timepoint, 'xk')
                    axes[i_ax].text(t_min_v, min_voltage_after_initial_timepoint, '  ttp: ' + str(round(t_min_v, 1)) +
                                    'ms, amp: ' + str(round(min_voltage_after_initial_timepoint - initial_voltage, 1)) +
                                    'mV', verticalalignment='bottom')
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_min_v,
                                                                    i_inital_timepoint, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axes[i_ax].plot(t_rise + find_max_min_from_ms, v_25_percent, 'xk')
                    axes[i_ax].text(t_rise + find_max_min_from_ms, v_25_percent,
                                    '  25% rt: ' + str(round(t_rise, 1)) + 'ms', verticalalignment='top',
                                    horizontalalignment='right')

        # set x lim and x label, show y-axis also on right and add legend
        axes[i_ax].set_xlim(0, info[0]['sim_time'])
        axes[i_ax].set_xlabel('t [ms]')
        axes[i_ax].tick_params(labelright=True)
        if b_legend:
            axes[i_ax].legend(title=info[0]['free_parameter_name'], prop={'size': 8})

    # disable x label for all but bottom subplot
    if not h_ax:
        [axes[n].set_xlabel(None) for n in range(n_gridrows - 1)]

    if not h_ax:
        return fig, axes
    else:
        return


def plot_spikes_mod(spikemons, info, t_const_event_marker=None, t_const_span_marker=None, dot_size=2,
                    b_black_background=False,
                    b_circuit_weight=True, b_circuit_delay=True, b_circuit_probability=True, b_circuit_idx=False):
    """Plot spiketimes of all model neurons with one subplot per population.

    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param dot_size: [default=2] size for spike markers in dot raster plot
    :type dot_size: int or float
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_circuit_weight: [default=True] plot synaptic weight values next to connecting lines in circuit diagram
    :type b_circuit_weight: bool
    :param b_circuit_delay: [default=True] plot synaptic delay values next to connecting lines
    :type b_circuit_delay: bool
    :param b_circuit_probability: [default=True] plot synaptic connection probability values next to connecting lines
    :type b_circuit_probability: bool
    :param b_circuit_idx: [default=True] plot synapse index next to connecting lines
    :type b_circuit_idx: bool
    :return:
        - fig: figure handle(s)
        - axes: axis handles, one sublist per figure
    :rtype:
        - fig: [matplotlib.figure.Figure]
        - axes: [[matplotlib.axes._subplots.AxesSubplot]]
    """

    # if spikemons and info contain a single object, convert them to lists
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(spikemons) is len(info), \
        "spikemons and info must have the same number of elements"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')
    fig = []
    axes = []
    n_figs = len(spikemons)
    n_gridcols = 5

    if n_figs > 20:
        print("WARNING: More than 20 figures are about to be created...")

    # loop through list of spike monitors
    for mon in range(n_figs):
        # create figure and set up grid for subplots
        n_populations = info[mon]['n_populations']
        n_gridrows = n_populations
        fig_tmp = plt.figure(figsize=(16, 9))
        fig.append(fig_tmp)
        plt.rcParams.update({'font.size': 12})
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
        axes.append([])
        color_map_dark = get_color_list(n_populations, color_map_in=C_COLORS)

        for pop in range(n_populations):
            # set up subplot
            axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (pop, 0), colspan=n_gridcols - 1, rowspan=1)
            axes[mon].append(axis_tmp)
            axes[mon][pop].set_ylabel(info[mon]['population_ids'][pop] + ' nrn_idx')
            if pop == 0:
                axes[mon][pop].set_title('Spike times of all neurons', weight='bold')
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axes[mon][pop].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            if t_const_span_marker is not None:
                axes[mon][pop].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            # go through neurons and plot all spikes for each neuron
            spiketimes = fcn.p_util.get_spiketimes_from_monitor(spikemons[mon], info[mon], pop)
            for nrn in range(len(spiketimes)):
                axes[mon][pop].plot(spiketimes[nrn], [nrn] * len(spiketimes[nrn]), '.',
                                    color=color_map_dark[pop], markersize=dot_size)
            # add right hand axis for population size and set y-axis limits
            axis_right = axes[mon][pop].twinx()
            axis_right.set_ylabel('n = ' + str(info[mon]['population_sizes'][pop]))
            axis_right.tick_params(which='both', right=False, labelright=False)
            axes[mon][pop].set_ylim(-0.5, info[mon]['population_sizes'][pop] - 0.5)

        # share x axis (time) among all horizontal plots
        [axes[mon][n].get_shared_x_axes().join(axes[mon][n], axes[mon][n + 1]) for n in range(n_gridrows - 1)]
        # disable x tick labels for voltage trace plots and autoscale all axes
        [axes[mon][n].set_xticklabels([]) for n in range(n_gridrows - 1)]

        # other settings
        axes[mon][0].set_xlim(0, info[mon]['sim_time'])
        axes[mon][n_gridrows - 1].set_xlabel('t [ms]')

        # plot network diagram in right grid column
        axis_net = plt.subplot2grid((n_gridrows, n_gridcols), (0, n_gridcols - 1), rowspan=n_gridrows, colspan=1)
        axes[mon].append(axis_net)
        plot_circuit_diagram(axis_net, info[mon], b_weight=b_circuit_weight, b_delay=b_circuit_delay,
                             b_probability=b_circuit_probability, b_syn_idx=b_circuit_idx)

    return fig, axes


def plot_burst_onsets(spiketimes_ms, offset_ms=0, t_const_event_marker=None):
    """

    :param spiketimes_ms: list of lists of lists of spike times of the neurons (sublist: neuron, subsublist: trial)
    :type spiketimes_ms: [[float]]
    :param offset_ms: [default=0] list of recording offsets, one per neuron
    :type offset_ms: list
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :return:
    """

    n_neurons = len(spiketimes_ms)
    if not isinstance(offset_ms, list):
        offset_ms = [offset_ms] * n_neurons

    n_gridrows = 6
    n_gridcols = 1
    fig = plt.figure(figsize=(24, 14))
    axes = []
    axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (0, 0), colspan=1, rowspan=5))
    axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (5, 0), colspan=1, rowspan=1))
    if t_const_event_marker is not None:
        axes[0].axvline(0, color=C_COLORS_GREY[0], linewidth=1, linestyle='--')
        axes[1].axvline(0, color=C_COLORS_GREY[0], linewidth=1, linestyle='--')

    # get times of first spike
    t_first_spike_nrn = []
    t_first_spike_avg = []
    t_first_spike_std = []
    for n, spikes_nrn in enumerate(spiketimes_ms):
        t_first_spike_nrn.append([])
        for t, spikes_trial in enumerate(spikes_nrn):
            # NOTE: hardcoded criterion to exclude post-call spikes (100ms after call onset)
            if spikes_trial and spikes_trial[0] < offset_ms[n] + 100:
                t_first_spike_nrn[n].append(spikes_trial[0])
        t_first_spike_avg.append(np.mean(t_first_spike_nrn[n]) - offset_ms[n])
        t_first_spike_std.append(np.std(t_first_spike_nrn[n]))
    idx_nrn_sort_by_tfs = np.argsort(t_first_spike_avg)

    # plot onsets of first spike for each neuron and trial
    row_t = -1  # row counter for trials from all neurons
    row_n = -1  # row counter for trials from all neurons
    cmap = matplotlib.cm.get_cmap('Dark2', n_neurons)
    for n, nrn in enumerate(idx_nrn_sort_by_tfs):
        row_n += 1
        spikes_nrn = spiketimes_ms[nrn]
        axes[0].axhline(row_t + 1, color=[.7, .7, .7], linewidth=1)
        for t, spikes_trial in enumerate(spikes_nrn):
            row_t += 1
            for s, spike in enumerate(spikes_trial):
                if t == 0 and s == 0:
                    label = 'neuron: ' + str(nrn + 1)
                else:
                    label = ''
                # plot all spikes of current trial in one row
                axes[0].plot((spike - offset_ms[nrn], spike - offset_ms[nrn]), (row_t, row_t + 1), linewidth=2,
                             color=cmap(nrn/n_neurons), label=label)
    # axes[0].set_xlim((-100, 150))
    axes[0].set_ylim((0, row_t + 1))
    axes[0].set_ylabel('all spikes per trial', fontsize=14)
    axes[0].tick_params(axis='both', labelsize=12)
    axes[0].legend(prop={'size': 12})

    # plot first spike of each trial in the same row for all trials of one neuron
    for n, nrn in enumerate(idx_nrn_sort_by_tfs):
        for first_spike_trial in t_first_spike_nrn[nrn]:
            axes[1].plot((first_spike_trial - offset_ms[nrn], first_spike_trial - offset_ms[nrn]), (n, n + 1),
                         linewidth=2, color=cmap(nrn / n_neurons))
    for n, nrn in enumerate(idx_nrn_sort_by_tfs):
        # plot the standard deviation
        axes[1].plot((t_first_spike_avg[nrn] - t_first_spike_std[nrn],
                      t_first_spike_avg[nrn] + t_first_spike_std[nrn]), (n + 0.5, n + 0.5), '-k',
                     linewidth=2)
        # plot the average
        axes[1].plot(t_first_spike_avg[nrn], n + 0.5, 'o', markeredgecolor='k',
                     markerfacecolor='w', markersize=6, markeredgewidth=2)
    # axes[1].set_xlim((-100, 150))
    axes[1].set_ylim((0, n_neurons))
    axes[1].set_xlabel('time from call onset [ms]', fontsize=14)
    axes[1].set_ylabel('first spike in each\ntrial per neuron', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=14)

    return fig, axes


def plot_t_first_spike_1d(spikemons, config, info, neuron_idx_abs, param_index_y, t_const_event_marker=None, dot_size=4,
                          idx_mons_colored=None, colors=None, param_unit=None, h_ax=None):
    """Plot the timepoint in the simulation when a neuron fires its first spike in dependence on the
    free parameter.

    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param neuron_idx_abs: absoluteindex of the neuron whose spike time is to be plotted
    :type neuron_idx_abs: int
    :param param_index_y: index to the values (synapse or population) of the free parameter - for y-axis in the plot
    :type param_index_y: int
    :param t_const_event_marker: [default=None] timepoint of an event in milliseconds. Marker will be drawn at 0 and
        all spiketimes will be negatively offset by this value.
    :type t_const_event_marker: int or float or None
    :param dot_size: size for points in plot
    :type dot_size: int or float
    :param idx_mons_colored: list of indices to monitors (runs) for which dots should be emphasized by color
    :type: list or None
    :param colors: list of color values for the emphasized monitors, same length as idx_mons_colored
    :type: list or None
    :param param_unit: brian2 unit (e.g. brian2.pA) in which to display the free parameter (y-axis). if None, use the
        unit as stated in config['parameter_units'].
    :type param_unit: brian2.units.fundamentalunits.Unit or None
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - axis: axis handle
    :rtype:
        - fig: matplotlib.figure.Figure
        - axis: matplotlib.axes._subplots.AxesSubplot
    """

    # if spikemons and info contain a single object (dictionary), convert them to lists
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(spikemons) is len(info), "statemons, spikemons and info must all have the same number of elements"
    assert type(param_index_y) is int, "param_index_y must be an integer"

    # if neuron_idx_abs is a negative number, translate that to a positive index
    if neuron_idx_abs < 0:
        neuron_idx_abs = sum(info[0]['population_sizes']) + neuron_idx_abs

    # get name of free parameter, stepsize and dict containing it e.g. "parameters_nrn"; if the config file contains one
    # restricted to one param atm!
    free_parameter_keys = list(config['free_parameter_stepsize'].keys())
    if not free_parameter_keys:
        free_parameter_name = None
        free_parameter_dict = None
    else:
        free_parameter_name = free_parameter_keys[0]
        if free_parameter_name in config['parameters_nrn']:
            free_parameter_dict = 'parameters_nrn'
        elif free_parameter_name in config['parameters_syn']:
            free_parameter_dict = 'parameters_syn'
        elif 'parameters_gen' in config and free_parameter_name in config['parameters_gen']:
            free_parameter_dict = 'parameters_gen'
        elif free_parameter_name in config['input_current']:
            free_parameter_dict = 'input_current'

    # get times of first spike
    spike_time = np.zeros(len(spikemons))
    y_values_run = []
    for mon in range(len(spikemons)):
        # get value of free parameter in this run. if a parameter unit was passed, apply that
        unitless_param_value = info[mon]['free_parameter_values'][param_index_y]
        # if this is a list (happens e.g. when free param is t_start), just use the last entry (quick fix)
        if isinstance(unitless_param_value, list):
            unitless_param_value = unitless_param_value[-1]
        if info[mon]['free_parameter_name'] in config['parameter_units']:
            original_unit = config['parameter_units'][info[mon]['free_parameter_name']]
        else:
            original_unit = ''
        if param_unit:
            if hasattr(b2.units, original_unit):
                y_values_run.append(unitless_param_value * eval('b2.' + original_unit) / param_unit)
                y_unit_str = str(param_unit)
            else:
                raise ValueError("plot_t_first_spike_1d(): Unrecognized free param unit '" + original_unit + "' in cfg")
        else:
            y_values_run.append(unitless_param_value)
            y_unit_str = original_unit
        idx_first_spike = np.where(spikemons[mon].i == neuron_idx_abs)[0]
        if np.any(spikemons[mon].t[idx_first_spike]):
            spike_time[mon] = spikemons[mon].t[idx_first_spike[0]] / b2.ms
    # set values NaN where no spike occured
    spike_time[spike_time == 0] = np.nan

    # plot
    if not h_ax:
        fig = plt.figure(figsize=(6, 4))
        axis = fig.add_subplot(1, 1, 1)
    else:
        axis = h_ax
    if t_const_event_marker is not None:
        axis.axvline(0, color=C_COLORS_GREY[0], linewidth=1, linestyle='--')
        spike_time = spike_time - t_const_event_marker
    i_color = 0
    if idx_mons_colored and not colors:
        colors = ['r'] * len(idx_mons_colored)
    for m in range(len(spike_time)):
        if idx_mons_colored and m in idx_mons_colored:
            axis.plot(spike_time[m], y_values_run[m], 'o', color=colors[i_color], markersize=dot_size)
            i_color += 1
        else:
            axis.plot(spike_time[m], y_values_run[m], 'ok', markersize=dot_size)

    # set axis ticks to parameter values, and other settings
    stepsize = abs(y_values_run[1] - y_values_run[0])
    axis.set_ylim(min(y_values_run) - stepsize, max(y_values_run) + stepsize)
    axis.set_xlabel('t first spike [ms]', weight='bold')
    axis.set_ylabel(free_parameter_name + ' (' + str(param_index_y) + ') [' + y_unit_str + ']', weight='bold')
    axis.xaxis.set_label_position('bottom')
    axis.xaxis.set_ticks_position('bottom')

    # set title (if synaptic parameter, include pre-synaptic and post-synaptic population ids in title)
    if free_parameter_dict == 'parameters_syn':
        axis.set_title('Time of first spike for neuron ' + str(neuron_idx_abs) + ' by ' + free_parameter_name +
                       ' (' + info[0]['population_ids'][info[0]['syn_pre_idx'][param_index_y]] + ' -> ' +
                       info[0]['population_ids'][info[0]['syn_post_idx'][param_index_y]] + ')', weight='bold')
        # if all free parameter values are negative, invert direction of y-axis
        if all([y_values_run[i] <= 0 for i in range(len(y_values_run))]):
            plt.gca().invert_yaxis()
    elif free_parameter_dict == 'parameters_nrn':
        axis.set_title('Time of first spike for neuron ' + str(neuron_idx_abs) + ' by ' + free_parameter_name +
                       ' (' + info[0]['population_ids'][param_index_y] + ')', weight='bold')
    else:
        axis.set_title('Time of first spike for neuron ' + str(neuron_idx_abs) + ' by ' +
                       free_parameter_name, weight='bold')

    if not h_ax:
        return fig, axis
    else:
        return


def plot_t_first_spike_1d_jamming(spikemons, config, info, neuron_idx_abs, param_index_y, idx_t_start,
                                  lat_playback_onset_to_t_start, t_const_event_marker=0, dot_size=4,
                                  marker_linestyle='--', idx_mons_colored=None, colors=None, h_ax=None):
    """Plot the timepoint in the simulation when a neuron fires its first spike in dependence on the
    free parameter. Compared to plot_t_first_spike_1d(), x- and y-axis are switched and x-axis is reversed. If,
    for example the free parameter is a timing parameter (e.g. input current t_start), this switches the relation, i.e.
    shows the delay of the neuron of interest spike from the point of view of its own spike relative to t_start (x-axis)
    instead of from the point of view of t_start in relation to its own spike.

    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param neuron_idx_abs: absolute index of the neuron whose spike time is to be plotted
    :type neuron_idx_abs: int
    :param param_index_y: index to the values (synapse or population) of the free parameter - for y-axis in the plot
    :type param_index_y: int
    :param idx_t_start: index to the value of t_start that is changing as a free parameter (e.g. start of ramp)
    :type idx_t_start: int
    :param lat_playback_onset_to_t_start: time in ms between supposed playback onset and t_start value at idx_t_start.
        adjusts x-axis values
    :type lat_playback_onset_to_t_start: int
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param dot_size: size for points in plot
    :type dot_size: int or float
    :param marker_linestyle: linestyle for event marker (default: '--' = dashed line)
    :type marker_linestyle: str
    :param idx_mons_colored: list of indices to monitors (runs) for which dots should be emphasized by color
    :type: list or None
    :param colors: list of color values for the emphasized monitors, same length as idx_mons_colored
    :type: list or None
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - axis: axis handle
    :rtype:
        - fig: matplotlib.figure.Figure
        - axis: matplotlib.axes._subplots.AxesSubplot
    """

    # if spikemons and info contain a single object (dictionary), convert them to lists
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(spikemons) is len(info), "statemons, spikemons and info must all have the same number of elements"
    assert type(param_index_y) is int, "param_index_y must be an integer"

    # if neuron_idx_abs is a negative number, translate that to a positive index
    if neuron_idx_abs < 0:
        neuron_idx_abs = sum(info[0]['population_sizes']) + neuron_idx_abs

    # get name of free parameter, stepsize and dict containing it e.g. "parameters_nrn"; if the config file contains one
    # restricted to one param atm!
    free_parameter_keys = list(config['free_parameter_stepsize'].keys())
    if not free_parameter_keys:
        free_parameter_name = None
        free_parameter_stepsize = None
        free_parameter_dict = None
    else:
        free_parameter_name = free_parameter_keys[0]
        free_parameter_stepsize = config['free_parameter_stepsize'][free_parameter_name]
        if free_parameter_name in config['parameters_nrn']:
            free_parameter_dict = 'parameters_nrn'
        elif free_parameter_name in config['parameters_syn']:
            free_parameter_dict = 'parameters_syn'
        elif 'parameters_gen' in config and free_parameter_name in config['parameters_gen']:
            free_parameter_dict = 'parameters_gen'
        elif free_parameter_name in config['input_current']:
            free_parameter_dict = 'input_current'

    # get bounds of values for free parameter
    lower_bounds = config[free_parameter_dict][free_parameter_name][0]
    upper_bounds = config[free_parameter_dict][free_parameter_name][1]
    lower_bound_y = lower_bounds[param_index_y]
    upper_bound_y = upper_bounds[param_index_y]

    # if the bounds contain a list, take the first value of that list (e.g. for input_current:t_start)
    if type(lower_bound_y) is list:
        lower_bound_y = lower_bound_y[0]
    if type(upper_bound_y) is list:
        upper_bound_y = upper_bound_y[0]

    # get all values of free parameter for x- and y-axis
    difference_y = upper_bound_y - lower_bound_y

    # get times of first spike
    spike_time = np.zeros(len(spikemons))
    y_values_run = []
    for mon in range(len(spikemons)):
        # get value of free parameter in this run
        if isinstance(info[mon]['free_parameter_values'][param_index_y], list):
            y_values_run.append(info[mon]['free_parameter_values'][param_index_y][idx_t_start])
        else:
            y_values_run.append(info[mon]['free_parameter_values'][param_index_y])
        idx_spikes = np.where(spikemons[mon].i == neuron_idx_abs)[0]
        # convert values back to milliseconds
        if np.any(spikemons[mon].t[idx_spikes]):
            spike_time[mon] = spikemons[mon].t[idx_spikes[0]] / b2.ms
    # set values NaN where no spike ocurred
    spike_time[spike_time == 0] = np.nan

    # time of first spike without inhibition (last entry, should be where inhibition ramp starts after first spike)
    t_first_spike_no_inhibition = spikemons[-1].t[np.where(spikemons[-1].i == neuron_idx_abs)[0][0]] / b2.ms

    # prepare plot
    if not h_ax:
        fig = plt.figure(figsize=(4.42, 2))
        axis = fig.add_subplot(1, 1, 1)
    else:
        axis = h_ax

    # plot markers, dots and crosses where the neuron didn't spike at all
    if t_const_event_marker is not None:
        axis.axhline(t_const_event_marker, color=C_COLORS_GREY[8], linewidth=1)
        axis.axvline(0, color=[0, 0, 0], linewidth=1, linestyle=marker_linestyle)
    axis.plot(t_first_spike_no_inhibition - np.array(y_values_run) + lat_playback_onset_to_t_start,
              spike_time - t_first_spike_no_inhibition, 'ok', markersize=dot_size)
    i_no_spike = np.where(np.isnan(spike_time))[0]
    axis.plot(t_first_spike_no_inhibition - np.array(y_values_run)[i_no_spike] + lat_playback_onset_to_t_start,
              [0] * len(i_no_spike), 'xk', markersize=dot_size)
    if idx_mons_colored:
        if not colors:
            colors = ['r'] * len(idx_mons_colored)
        for i, m in enumerate(idx_mons_colored):
            if np.isnan(spike_time[m]):
                axis.plot(t_first_spike_no_inhibition - y_values_run[m] + lat_playback_onset_to_t_start,
                          0, 'x', color=colors[i], markersize=dot_size)
            else:
                axis.plot(t_first_spike_no_inhibition - y_values_run[m] + lat_playback_onset_to_t_start,
                          spike_time[m] - t_first_spike_no_inhibition, 'o', color=colors[i], markersize=dot_size)

    # set axis ticks to parameter values, and other settings
    # stepsize = abs(y_values_run[1] - y_values_run[0])
    # axis.set_xlim(min(y_values_run) - stepsize, max(y_values_run) + stepsize)
    axis.set_xlabel('PM burst relative to playback onset [ms]', weight='regular')
    axis.set_ylabel('PM burst delay [ms]', weight='regular')
    axis.xaxis.set_label_position('bottom')
    axis.xaxis.set_ticks_position('bottom')

    if not h_ax:
        fig.tight_layout()
        return fig, axis
    else:
        return


def plot_connection_matrix(connectivity, info):
    """Matrix plots of the synaptic connections between the neurons of two populations (one figure per population pair)

    :param connectivity: dictionary or list of dictionaries containing synaptic connectivity information
        (one dict per connection between populations)
    :type connectivity: dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (only first dict used)
    :type info: dict or list
    :return: fig: list of figure handles
    :rtype: [matplotlib.figure.Figure]
    """

    # if connectivity and info contain a single object (dictionary), convert them to lists
    if type(connectivity) is not list:
        connectivity = [connectivity]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, dict) for item in connectivity), \
        "connectivity must be a dictionary or list of dictionaries"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"

    fig = []
    for c in range(len(connectivity)):
        # construct connection matrix from synapse indices (i = pre, j = post)
        idx_pre_pop = info[0]['syn_pre_idx'][c]
        idx_post_pop = info[0]['syn_post_idx'][c]
        pop_sz_pre = info[0]['population_sizes'][idx_pre_pop]
        pop_sz_post = info[0]['population_sizes'][idx_post_pop]
        connection_matrix = np.zeros((pop_sz_pre, pop_sz_post))
        connection_matrix[connectivity[c]['i'], connectivity[c]['j']] = 1
        # set up figure and plot matrix as pixel-perfect figimage
        dpi = 80
        y_pixels = pop_sz_pre
        x_pixels = pop_sz_post
        fig.append(plt.figure(figsize=(x_pixels/dpi, y_pixels/dpi), dpi=dpi))
        fig[c].figimage(connection_matrix, cmap='cividis', norm=matplotlib.colors.Normalize(0, 1))

    return fig


def plot_connection_matrix_generator(connectivity_generator, info):
    """Matrix plots of the synaptic connections between the neurons of two populations (one figure per population pair)

    :param connectivity_generator: dictionary or list of dictionaries containing spikegenerator connectivity information
        (one dict per connection between spikegenerator and population)
    :type connectivity_generator: dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (only first dict used)
    :type info: dict or list
    :return: fig: list of figure handles
    :rtype: [matplotlib.figure.Figure]
    """

    # todo: does probably not work with unequal generator and population sizes. likely unnecessary anyway due to
    # (todo cont) selective loading of spiketimes thorugh p_input.load_spiketimes_for_gen

    # if connectivity_generator and info contain a single object (dictionary), convert them to lists
    if type(connectivity_generator) is not list:
        connectivity_generator = [connectivity_generator]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, dict) for item in connectivity_generator), \
        "connectivity_generator must be a dictionary or list of dictionaries"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"

    fig = []
    for c in range(len(connectivity_generator)):
        # construct connection matrix from synapse indices (i = pre, j = post)
        pop_sz_pre = info[0]['generator_sizes'][c]
        pop_sz_post = info[0]['population_sizes'][c]
        connection_matrix = np.zeros((pop_sz_pre, pop_sz_post))
        connection_matrix[connectivity_generator[c]['i'], connectivity_generator[c]['j']] = 1
        # set up figure and plot matrix as pixel-perfect figimage
        dpi = 80.
        y_pixels = pop_sz_pre
        x_pixels = pop_sz_post
        fig.append(plt.figure(figsize=(x_pixels/dpi, y_pixels/dpi), dpi=int(dpi)))
        fig[c].figimage(connection_matrix, cmap='cividis')

    return fig


def plot_sensitivity_traces(statemons, spikemons, infos, config_lo, conditions_met, x_values, y_values, baseline_cond,
                            t_baseline, t_spikes, xlabel=None, ylabel=None, xlims=None, figsize=(16, 9)):
    """Generate 2d matrix of subplots of traces in parameter space of two varied parameters, that visualizes where
    certain conditions are met. Run p_analysis.check_conditions() first.

    :param statemons: list of dict(s) containing StateMonitor data from b2...get_states()
    :type statemons: list
    :param spikemons: list of dict(s) containing SpikeMonitor data from b2...get_states()
    :type spikemons: list
    :param infos: list of dict(s) containing additional information about simulation (as created by run_simulation())
    :type infos: list
    :param config_lo: config dict as loaded from .json file (output directory), contains lower limits of all params
    :type config_lo: dict
    :param conditions_met: numpy array of dimension c*x*y as returned by p_analysis.check_conditions()
    :type conditions_met: numpy.ndarray
    :param x_values: values of the first varied parameter (usually last parameter in cfg list), used for x-tick-labels
    :type x_values: list
    :param y_values: values of the second varied parameter (usually first parameter in cfg list), used for y-tick-labels
    :type y_values: list
    :param baseline_cond: condition for baseline membrane potential. tuple of lower and upper limit for condition to
        be considered met.
    :type baseline_cond: tuple or list
    :param t_baseline: time window in which the baseline membrane potential condition was checked (start, end)
    :type t_baseline: tuple or list
    :param t_spikes: time window in which the number of spikes condition was checked (start, end)
    :type t_spikes: tuple or list
    :param xlabel: x-axis label
    :type xlabel: str
    :param ylabel: y-axis label
    :type ylabel: str
    :param xlims: x-axis limits (lower, upper)
    :type xlims: tuple or list
    :param figsize: size of figure in inches (width, height)
    :type figsize: tuple or list
    :return:
        - fig: figure handle
        - axes: list of axis handles
    :rtype:
        - fig: matplotlib.figure.Figure
        - axes: [matplotlib.axes._subplots.AxesSubplot]
    """

    n_gridcols = conditions_met.shape[2]
    n_gridrows = conditions_met.shape[1]
    matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    fig = plt.figure(figsize=figsize)

    # loop through sims and plot trace
    axes = []
    i = 0
    for y in range(n_gridrows):
        for x in range(n_gridcols):
            # get index to neuron of interest
            nrn_oi_abs = p_util.get_abs_from_rel_nrn_idx(config_lo['plot']['idx_nrn_oi_relative'],
                                                         config_lo['plot']['idx_pop_oi'],
                                                         infos[i]['population_sizes'])
            axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (y, x)))
            # plot rectangle for number of spikes condition
            if conditions_met[0][y][x]:
                axes[i].axvspan(t_spikes[0] - config_lo['misc']['playback_start'],
                                t_spikes[1] - config_lo['misc']['playback_start'], color=(.8, 1, .8))
            else:
                axes[i].axvspan(t_spikes[0] - config_lo['misc']['playback_start'],
                                t_spikes[1] - config_lo['misc']['playback_start'], color=(1, .8, .8))
            # plot rectangle for baseline membrane potential condition
            if conditions_met[1][y][x]:
                axes[i].add_patch(plt.Rectangle((t_baseline[0] - config_lo['misc']['playback_start'], baseline_cond[0]),
                                                t_baseline[1] - t_baseline[0],
                                                baseline_cond[1] - baseline_cond[0], color=(.8, 1, .8)))
            else:
                axes[i].add_patch(plt.Rectangle((t_baseline[0] - config_lo['misc']['playback_start'], baseline_cond[0]),
                                                t_baseline[1] - t_baseline[0],
                                                baseline_cond[1] - baseline_cond[0], color=(1, .8, .8)))
            # plot trace
            plt.plot(statemons[i].t / b2.ms - config_lo['misc']['playback_start'],
                     statemons[i].v[nrn_oi_abs, :] / b2.mV, color=[0, 0, 0], linewidth=1, label=str(i))
            # plot artificial spikes
            spiketimes = spikemons[i].t[spikemons[i].i == nrn_oi_abs] / b2.ms - config_lo['misc']['playback_start']
            thresh = infos[i]['v_thresh'][nrn_oi_abs]
            for s in range(len(spiketimes)):
                plt.plot([spiketimes[s], spiketimes[s]], [thresh, thresh + 30], color=[0, 0, 0], linewidth=1)

            # add parameter values as axis labels
            axes[i].tick_params(length=2)
            if y == 0:
                axes[i].xaxis.set_ticklabels([])
                ax_r = axes[i].twiny()
                ax_r.xaxis.set_ticklabels([])
                ax_r.set_xticks([])
                if x == round(n_gridcols/2) - 1:
                    ax_r.set_xlabel(xlabel + '\n\n' + str(x_values[x]))
                else:
                    ax_r.set_xlabel(str(x_values[x]))
            if not y == n_gridrows - 1:
                axes[i].xaxis.set_ticklabels([])
            else:
                axes[i].set_xlabel(' ')
                if x == round(n_gridcols/2) - 1:
                    axes[i].set_xlabel(axes[i].get_xlabel() + '\n' + 'Time from call production onset [ms]')
            if x == n_gridcols - 1:
                axes[i].yaxis.set_ticklabels([])
                ax_r = axes[i].twinx()
                ax_r.yaxis.set_ticklabels([])
                ax_r.set_yticks([])
                if y == round(n_gridrows / 2) - 1:
                    ax_r.set_ylabel(str(y_values[y]) + '\n\n' + ylabel)
                else:
                    ax_r.set_ylabel(str(y_values[y]))
            if not x == 0:
                axes[i].yaxis.set_ticklabels([])
            else:
                axes[i].set_ylabel(' ')
                if y == round(n_gridrows/2) - 1:
                    axes[i].set_ylabel('$\mathrm{V_m}$ [mV]' + '\n' + axes[i].get_ylabel())
            i += 1

    # share y-axes
    [axes[n].get_shared_x_axes().join(axes[n], axes[n + 1]) for n in range(len(axes) - 1)]
    [axes[n].get_shared_y_axes().join(axes[n], axes[n + 1]) for n in range(len(axes) - 1)]
    if xlims:
        axes[0].set_xlim([xl - config_lo['misc']['playback_start'] for xl in xlims])
    else:
        axes[0].set_xlim([0, statemons[0].t[-1] / b2.ms])
    axes[0].autoscale(axis='y')

    return fig, axes


def plot_circuit_diagram(axis, info_dict, b_weight=True, b_delay=True, b_probability=True, b_syn_idx=False,
                         line_colors=tuple(np.asarray(np.ones(128) * 1, dtype=int))):
    """Plot a circuit diagram with each neuron population (in info_dict) represented by a circle and each synaptic
    connection represented as lines between them with triangles for postive synaptic weights (excitatory) and circles
    for negative (inhibitory)

    :param axis: pyplot axis handle in which the circuit diagram should be plotted
    :type axis: matplotlib.axes._subplots.AxesSubplot
    :param info_dict: dictionary as created by the simulation (e.g. lif.py). Do not pass the list that is saved by
        the simulation, but only a single element (i.e. dict).
    :type info_dict: dict
    :param b_weight: [default=True] plot synaptic weight values next to connecting lines
    :type b_weight: bool
    :param b_delay: [default=True] plot synaptic delay values next to connecting lines
    :type b_delay: bool
    :param b_probability: [default=True] plot synaptic connection probability values next to connecting lines
    :type b_probability: bool
    :param b_syn_idx: [default=False] plot synapse index next to connecting lines
    :type b_syn_idx: bool
    :param line_colors: [default=(1,1,...)] tuple of length==n_synapses, consisting of an index to C_COLORS_GREY
        (defined in this file) for each synpase, setting the color for the respective connection line. 1 == black
    :type line_colors: tuple
    :return:
    """

    # get number of populations (i.e. circles in the diagram; can be single neurons)
    n_populations = info_dict['n_populations']
    axis.set_ylim(n_populations - 0.5, -0.5)
    circle_x = [0.2, 0.8] * np.ceil(n_populations / 2).astype(int)
    circle_radius = 0.15
    in_syn_radius = 0.04  # radius of circle representing inhibitory synapse
    recurrent_radius = 0.11  # radius of circle representing recurrent connections (population synapsing onto itself)
    color_map_dark = get_color_list(n_populations + 3, color_map_in=C_COLORS)
    color_map_grey = get_color_list(3, color_map_in=C_COLORS_GREY)

    for pop in range(n_populations):
        # plot a circle representing each population next to its voltage trace (circle y = index of pop: 0, 1, ...)
        circle_tmp = plt.Circle((circle_x[pop], pop), circle_radius, edgecolor=color_map_dark[pop], facecolor='w',
                                linewidth=2, zorder=2)
        axis.add_artist(circle_tmp)
        plt.text(circle_x[pop], pop, info_dict['population_ids'][pop][0], fontsize=16, zorder=3,
                 horizontalalignment='center', verticalalignment='center')
        # plot population name
        plt.text(1.2, pop, info_dict['population_ids'][pop], rotation='vertical', verticalalignment='center')
        # loop through indices of the synapses in which the population is the pre-synaptic population
        i_pop_is_pre = [i for i, p in enumerate(info_dict['syn_pre_idx']) if p == pop]
        for syn in i_pop_is_pre:
            post = info_dict['syn_post_idx'][syn]  # index of post-synaptic population at current synapse
            # check if the synapse connects the population to itself; if so, plot circle to represent connection
            if post == pop:  # connection from one population onto itself
                circle_tmp = plt.Circle((circle_x[pop] + circle_radius*1.5, pop), recurrent_radius,
                                        edgecolor=color_map_grey[line_colors[syn]], facecolor='w', linewidth=1, zorder=1)
                axis.add_artist(circle_tmp)
                # add text to line displaying synaptic weight and delay as well as legend text on bottom of axis
                next_to_nrn = [circle_x[pop] + circle_radius + recurrent_radius*1.8, pop]
                if b_weight:
                    plt.text(next_to_nrn[0], next_to_nrn[1], str(round(info_dict['syn_weight'][syn], 3)), zorder=2,
                             horizontalalignment='left', verticalalignment='bottom', color=color_map_dark[-3], size=9)
                if b_delay:
                    plt.text(next_to_nrn[0], next_to_nrn[1], '\n' + str(info_dict['syn_delay'][syn]), zorder=2,
                             horizontalalignment='left', verticalalignment='center', color=color_map_dark[-2], size=9)
                if b_probability and info_dict['connection_probability'][syn] < 1:
                    plt.text(next_to_nrn[0], next_to_nrn[1], '\n\n\n' + str(info_dict['connection_probability'][syn]),
                             zorder=2, horizontalalignment='left', verticalalignment='center', color=color_map_dark[-1],
                             size=9)
                # add text for synaptic index next to line
                if b_syn_idx:
                    plt.text(next_to_nrn[0], next_to_nrn[1], str(syn), zorder=2, horizontalalignment='left',
                             verticalalignment='bottom', color=color_map_grey[line_colors[syn]])
                # plot an arrow (head) to represent an excitatory synapse, a circle to represent an inhibitory synapse
                if info_dict['syn_weight'][syn] > 0:
                    arrow_length = [circle_radius * np.cos(.48), circle_radius * np.sin(.48)]
                    plt.arrow(circle_x[post], post, arrow_length[0] * 1.05, arrow_length[1] * 1.05,
                              head_width=0.08, head_length=0.08, facecolor='k', zorder=1)
                elif info_dict['syn_weight'][syn] < 0:
                    circle_dist = [(circle_radius + in_syn_radius) * np.cos(.48),
                                   (circle_radius + in_syn_radius) * np.sin(.48)]
                    circle_in_syn = plt.Circle(
                        (circle_x[post] + circle_dist[0] * 1.05, post + circle_dist[1] * 1.05),
                        in_syn_radius, edgecolor='k', facecolor='w', linewidth=2, zorder=1)
                    axis.add_artist(circle_in_syn)
            else:  # connections between two populations
                # plot a line connecting circles for each synapse
                plt.plot([circle_x[pop], circle_x[post]], [pop, post], color=color_map_grey[line_colors[syn]], zorder=1)
                # get angle of line connecting populations
                angle_rad = np.arctan2(pop - post, circle_x[pop] - circle_x[post])
                angle_deg = np.rad2deg(angle_rad)
                # add text to line displaying synaptic weight and delay as well as legend text on bottom of axis
                between_nrns = [circle_x[pop] + (circle_x[post] - circle_x[pop]) * 2/3, pop + (post - pop) * 2/3]
                if b_weight:
                    plt.text(between_nrns[0], between_nrns[1], str(round(info_dict['syn_weight'][syn], 3)), zorder=2,
                             rotation=-angle_deg, rotation_mode='anchor', horizontalalignment='center',
                             verticalalignment='bottom', color=color_map_dark[-3], size=9)
                if b_delay:
                    plt.text(between_nrns[0], between_nrns[1], '\n' + str(info_dict['syn_delay'][syn]), zorder=2,
                             rotation=-angle_deg, rotation_mode='anchor', horizontalalignment='center',
                             verticalalignment='center', color=color_map_dark[-2], size=9)
                if b_probability and info_dict['connection_probability'][syn] < 1:
                    plt.text(between_nrns[0], between_nrns[1], '\n\n\n' + str(info_dict['connection_probability'][syn]),
                             zorder=2, rotation=-angle_deg, rotation_mode='anchor', horizontalalignment='center',
                             verticalalignment='center', color=color_map_dark[-1], size=9)
                # add text for synaptic index next to line
                if b_syn_idx:
                    plt.text(between_nrns[0], between_nrns[1], str(syn), zorder=2,
                             rotation=-angle_deg, rotation_mode='anchor', horizontalalignment='center',
                             verticalalignment='bottom', color=color_map_grey[line_colors[syn]])
                # plot an arrow (head) to represent an excitatory synapse, a circle to represent an inhibitory synapse
                if info_dict['syn_weight'][syn] > 0:
                    arrow_length = [circle_radius * np.cos(angle_rad), circle_radius * np.sin(angle_rad)]
                    plt.arrow(circle_x[post], post, arrow_length[0] * 1.05, arrow_length[1] * 1.05,
                              head_width=0.08, head_length=0.08, facecolor='k', zorder=1)
                elif info_dict['syn_weight'][syn] < 0:
                    circle_dist = [(circle_radius + in_syn_radius) * np.cos(angle_rad),
                                   (circle_radius + in_syn_radius) * np.sin(angle_rad)]
                    circle_in_syn = plt.Circle((circle_x[post] + circle_dist[0] * 1.05, post + circle_dist[1] * 1.05),
                                               in_syn_radius, edgecolor='k', facecolor='w', linewidth=2, zorder=1)
                    axis.add_artist(circle_in_syn)

    # plot legend text
    if b_weight:
        plt.text(.5, n_populations - 1 + .4, 'synaptic weight [mV]', horizontalalignment='center',
                 color=color_map_dark[-3], size=9)
    if b_delay:
        plt.text(.5, n_populations - 1 + .55, 'synaptic delay [ms]', horizontalalignment='center',
                 color=color_map_dark[-2], size=9)
    if b_probability:
        plt.text(.5, n_populations - 1 + .7, 'connection probability', horizontalalignment='center',
                 color=color_map_dark[-1], size=9)

    # set net axis aspect equal (i.e. resizing figure will not affect shape of axis)
    axis.set_aspect('equal')
    axis.set_xlim(0, 1)
    axis.axis('off')


def plot_psths(t_values, psths=None, psths_smooth=None, labels=None, t_const_event_marker=None,
               t_const_span_marker=None, marker_linestyle='--', colors=None, linewidth=None, b_black_background=False,
               b_plot_average=False, t_baseline=None, t_peak_range=None, b_normalize_rate=False, b_markers=False,
               b_legend=True, h_ax=None):
    """Plot multiple peri-stimulus time histograms (psth) into a single figure.

    :param t_values: (list of) list(s) of time values of psth bins (left value), as returned by p_analysis.get_psth()
    :type t_values: [float] or [[float]]
    :param psths: [default=None] (list of) psth(s) as returned by p_analysis.get_psth(). pass either psths,
        psths_smooth or both
    :type psths: [float] or [[float]] or None
    :param psths_smooth: [default=None] list of smoothed psth as returned by p_analysis.get_psth()
    :type psths_smooth: [float] or [[float]] or None
    :param labels: [default=None] (list of) string(s) of labels for the different psths (e.g. neurons). Set entries
        to None to exclude them from the legend (e.g.: ['bla', None, 'bla'])
    :type labels: str or list or None
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param marker_linestyle: linestyle for event marker (default: '--' = dashed line)
    :type marker_linestyle: str
    :param colors: list of color values, one color for each psth
    :type colors: list or None
    :param linewidth: (list of) linewidth values, one color for each psth
    :type linewidth: list or None
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_plot_average: [default=False] plot the average of all psths
    :type b_plot_average: bool
    :param t_baseline: time interval [beginning, end] during which to calculate average baseline activity.
    :type t_baseline: list or tuple or None
    :param t_peak_range: time interval [beginning, end] in which to detect peak activity.
    :type t_peak_range: list or tuple or None
    :param b_normalize_rate: if True, psths will be normalized so that baseline activity is 1.
    :type b_normalize_rate: bool
    :param b_markers: if True, additional markers will be plotted (baseline, peak, ...)
    :type b_markers: bool
    :param b_legend: if True, add legend
    :type b_legend: bool
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - axis: axis handle
    :rtype:
        - fig: matplotlib.figure.Figure
        - axis: matplotlib.axes._subplots.AxesSubplot
    """

    # check inputs
    assert type(t_values) is list, "t_values must be a (list of) list(s)"
    assert type(psths) is list or psths is None, "psths must be a (list of) list(s)"
    assert type(psths_smooth) is list or psths_smooth is None, "psths_smooth must be a (list of) list(s)"
    assert type(labels) is list or labels is None or type(labels) is str,  "labels must be a (list of) string(s)"
    # if t_values, psths and/or psths_smooth are a single list, convert them to a list of a list
    if type(t_values[0]) is not list:
        t_values = [t_values]
    if psths is not None and type(psths[0]) is not list:
        psths = [psths]
    if psths_smooth is not None and type(psths_smooth[0]) is not list:
        psths_smooth = [psths_smooth]
    if type(labels) is str:
        labels = [labels]
    if b_normalize_rate:
        assert t_baseline and len(t_baseline) == 2, "b_normalize_rate is True, but t_baseline not given"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')

    # create figure and axis
    if not h_ax:
        fig = plt.figure(figsize=(16, 9))
        plt.rcParams.update({'font.size': 12})
        axis = plt.axes()
    else:
        axis = h_ax
    if psths_smooth and not psths:
        axis.set_title('Peri-stimulus-time histogram (smoothed)', weight='bold')
    else:
        axis.set_title('Peri-stimulus-time histogram', weight='bold')
    if b_normalize_rate:
        axis.set_ylabel('avg. spike rate rel. to baseline')
    else:
        axis.set_ylabel('average spike rate [Hz]')

    # if no labels were passed, set arbitrary labels
    if not labels:
        labels = [str(n) for n in range(len(t_values))]

    # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
    if t_const_event_marker is not None:
        axis.axvline(t_const_event_marker, color=[0, 0, 0], linewidth=1, linestyle=marker_linestyle)
    if t_const_span_marker is not None:
        axis.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[8])

    # get default color cycle and duplicate each entry for psth and psth_smooth if both will be printed
    if psths and psths_smooth:
        default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        idx_dupl_color_cycle = np.repeat(list(range(len(default_color_cycle))), 2).tolist()
        axis.set_prop_cycle(color=[default_color_cycle[idx_dupl_color_cycle[val]]
                                   for val in range(len(idx_dupl_color_cycle))])

    # get baseline interval in samples
    if t_baseline is not None:
        samp_baseline_start = np.where(np.array(t_values[0]) >= t_baseline[0])[0][0]
        samp_baseline_end = np.where(np.array(t_values[0]) <= t_baseline[1])[0][-1]

    # plot psths
    min_x = 0
    max_x = 0
    if psths_smooth and not psths:
        linestyle_smoothed = '-'
    else:
        linestyle_smoothed = '--'
    for idx in range(len(t_values)):
        if colors and len(colors) >= len(t_values):
            color_cur = colors[idx]
        else:
            color_cur = plt.cm.tab20(idx)
        if linewidth and len(linewidth) >= len(t_values):
            linewidth_cur = linewidth[idx]
        else:
            linewidth_cur = 2
        if psths:
            # get baseline activity
            if t_baseline is not None:
                baseline_activity = np.mean(psths[idx][samp_baseline_start:samp_baseline_end])
                if b_markers:
                    # mark peak
                    if not t_peak_range:
                        t_peak_range_idx = (0, len(t_values))
                    else:
                        t_peak_range_idx = (np.where(np.array(t_values[idx])>=t_peak_range[0])[0][0],
                                          np.where(np.array(t_values[idx])>=t_peak_range[1])[0][0])
                    peak_y = np.max(psths[idx][t_peak_range_idx[0]:t_peak_range_idx[1]])
                    peak_x = t_values[idx][np.argmax(psths[idx][t_peak_range_idx[0]:t_peak_range_idx[1]]) \
                                           + t_peak_range_idx[0]]
                    axis.plot(peak_x, peak_y, 'o', markersize=8)
                    axis.text(peak_x, peak_y, 't: ' + str(round(peak_x, 1)) + 'ms\nrate: ' + str(round(peak_y, 1)))
                    # mark baseline
                    axis.axvspan(*t_baseline, color=(.9, .9, .9))
                    axis.axhline(np.mean(baseline_activity), linestyle='--', color=color_cur)
                    # mark return to baseline
            if b_normalize_rate:
                axis.plot(t_values[idx], psths[idx] / baseline_activity, color=color_cur, label=labels[idx],
                          linewidth=linewidth_cur)
            else:
                axis.plot(t_values[idx], psths[idx], color=color_cur, label=labels[idx],
                          linewidth=linewidth_cur)
        if psths_smooth:
            # get baseline activity
            if t_baseline is not None:
                baseline_activity = np.mean(psths_smooth[idx][samp_baseline_start:samp_baseline_end])
                if b_markers:
                    # mark peak
                    if not t_peak_range:
                        t_peak_range_idx = (0, len(t_values))
                    else:
                        t_peak_range_idx = (np.where(np.array(t_values[idx]) >= t_peak_range[0])[0][0],
                                            np.where(np.array(t_values[idx]) >= t_peak_range[1])[0][0])
                    peak_y = np.max(psths_smooth[idx][t_peak_range_idx[0]:t_peak_range_idx[1]])
                    peak_i = np.argmax(psths_smooth[idx][t_peak_range_idx[0]:t_peak_range_idx[1]]) + t_peak_range_idx[0]
                    peak_x = t_values[idx][peak_i]
                    axis.plot(peak_x, peak_y, 'o', color=color_cur, markersize=8)
                    axis.text(peak_x, peak_y, str(round(peak_x, 1)) + 'ms\n ' + str(round(peak_y, 1)))
                    # mark baseline
                    axis.axvspan(*t_baseline, color=(.9, .9, .9))
                    axis.axhline(np.mean(baseline_activity), linestyle='--', color=color_cur)
                    # mark return to baseline
                    first_below_bl_i = np.where(np.array(psths_smooth[idx][peak_i:]) <= baseline_activity)[0][0] + peak_i
                    first_below_bl_x = t_values[idx][first_below_bl_i]  # first point below baseline
                    diff_to_prev_y = psths_smooth[idx][first_below_bl_i] - psths_smooth[idx][first_below_bl_i - 1]
                    diff_to_bl_y = psths_smooth[idx][first_below_bl_i] - baseline_activity
                    ratio_diff = diff_to_bl_y / diff_to_prev_y
                    diff_to_prev_x = t_values[idx][first_below_bl_i] - t_values[idx][first_below_bl_i - 1]
                    diff_to_bl_x = diff_to_prev_x * ratio_diff
                    bl_crossing = first_below_bl_x - diff_to_bl_x
                    axis.plot(bl_crossing, baseline_activity, 'o', color=color_cur, markersize=8, fillstyle='none')
                    axis.text(bl_crossing, baseline_activity, str(round(bl_crossing, 1)) + 'ms\n '
                              + str(round(baseline_activity, 1)))
            if b_normalize_rate:
                axis.plot(t_values[idx], psths_smooth[idx] / baseline_activity, linestyle_smoothed, color=color_cur,
                          label=labels[idx], linewidth=linewidth_cur)
            else:
                axis.plot(t_values[idx], psths_smooth[idx], linestyle_smoothed, color=color_cur,
                          label=labels[idx], linewidth=linewidth_cur)
        if np.min(t_values[idx]) < min_x:
            min_x = np.min(t_values[idx])
        if np.max(t_values[idx]) > max_x:
            max_x = np.max(t_values[idx])

    # plot average psth
    if b_plot_average:
        axis.plot()

    # other settings
    axis.set_xlim(min_x, max_x)
    axis.set_ylim(0, axis.get_ylim()[1])
    axis.set_xlabel('t [ms]')
    if b_legend:
        axis.legend(prop={'size': 9})

    if not h_ax:
        return fig, axis
    else:
        return


def plot_spikes_psth(spiketimes_ms, duration_ms, dot_size=2, recording_offset_ms=0, psth_dt_ms=5, plot_duration=None,
                     t_const_event_marker=None, t_const_span_marker=None, b_black_background=False,
                     b_smoothed_psth=True):
    """Plot all spiketimes of one or more neurons/trials of one neuron into one subplot, with the
    peri-stimulus time histogram in a separate subplot.

    :param spiketimes_ms: list of lists of spike times of the neurons (one sublist per neuron or trial)
    :type spiketimes_ms: [[float]]
    :param duration_ms: total duration of recording / simulation and therefore of psth
    :type duration_ms: int or float
    :param dot_size: [default=2] size for spike markers in dot raster plot
    :type dot_size: int or float
    :param recording_offset_ms: [default=0] plots of recording data will start at this timepoint into the recording [ms]
    :type recording_offset_ms: int or float
    :param psth_dt_ms: [default=5] time bin [ms] for the peri-stimulus-time histogram (psth)
    :type psth_dt_ms: int or float
    :param plot_duration: [default=None] if not None, this determins the right hand x-axis limit for the plots
    :type plot_duration: int or float or None
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_smoothed_psth: [default=True] also plot smoothed psth as dotted line
    :type b_smoothed_psth: bool
    :return:
        - fig: figure handle
        - axes: list of axis handles
        - t_values: list of time values of psth bins (left edges) in milliseconds as returned by p_analysis.get_psth()
        - psth: list of estimated spike rates per time bin as returned by p_analysis.get_psth()
        - psth_smooth: same as psth, but smoothed using Savitzky-Golay filter (None if b_smoothed_psth==False)
    :rtype:
        - fig: matplotlib.figure.Figure
        - axes: [matplotlib.axes._subplots.AxesSubplot]
        - t_values: [float]
        - psth: [float]
        - psth_smooth: [float] or None
    """

    # check inputs
    assert type(spiketimes_ms) is list, \
        "spiketimes_ms must be a list of lists of spiketimes (in milliseconds; one sublist per trace)"
    assert all(isinstance(item, list) for item in spiketimes_ms), \
        "spiketimes_ms must be a list of lists of spiketimes (in milliseconds; one sublist per trace)"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')
    axes = []
    n_gridcols = 1

    # create figure and set up grid for subplots
    n_gridrows = 2
    fig = plt.figure(figsize=(16, 9))
    plt.rcParams.update({'font.size': 12})
    matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)

    # plot all spikes
    axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (0, 0), colspan=n_gridcols, rowspan=1)
    axes.append(axis_tmp)
    axes[0].set_ylabel('[Trial/Neuron] #')
    # add right hand axis for population size
    axis_right = axes[0].twinx()
    axes[0].set_title('Spike times', weight='bold')
    # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
    if t_const_event_marker is not None:
        axes[0].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
    if t_const_span_marker is not None:
        axes[0].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
        axes[0].text(t_const_span_marker[0] + (t_const_span_marker[1] - t_const_span_marker[0]) / 2, 1, "playback",
                     color=C_COLORS_GREY[6], zorder=1, verticalalignment='bottom', horizontalalignment='center')
    # go through list entries of spiketimes_ms (neurons or trials) and plot all spikes for each
    spiketimes_offset = []
    for nrn in range(len(spiketimes_ms)):
        # subtract offset from time values
        spiketimes_offset.append([spiketimes_ms[nrn][idx] - recording_offset_ms
                                  for idx in range(len(spiketimes_ms[nrn]))])
        axes[0].plot(spiketimes_offset[nrn], [nrn] * len(spiketimes_offset[nrn]), '.',
                     color='black', markersize=dot_size)
    # set y-axis limits and sample size for right axis
    axes[0].set_ylim(-0.5, len(spiketimes_ms) - 0.5)
    axis_right.set_ylabel('n = ' + str(len(spiketimes_ms)))
    axis_right.tick_params(which='both', right=False, labelright=False)

    # calculate peri-stimulus-time histogram (psth)
    t_values, psth, psth_smooth = p_analysis.get_psth(spiketimes_offset, duration_ms, psth_dt_ms=psth_dt_ms,
                                                      b_smoothed_psth=b_smoothed_psth)

    # set maximum plot time for axis limits
    if plot_duration:
        t_max = plot_duration
    else:
        t_max = max(t_values)

    # set up subplot for psths
    axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (1, 0), colspan=n_gridcols, rowspan=1)
    axis_tmp.set_title('Peri-stimulus-time histogram', weight='bold')
    axis_tmp.set_ylabel('average spike rate [Hz]')
    # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
    if t_const_event_marker is not None:
        axis_tmp.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
    if t_const_span_marker is not None:
        axis_tmp.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
    # plot psths
    axis_tmp.plot(t_values, psth, color='black', label='psth')
    if b_smoothed_psth:
        axis_tmp.plot(t_values, psth_smooth, '--', color=C_COLORS_GREY[4], label='smoothed psth')
    axes.append(axis_tmp)
    # add right hand axis for bin size text
    axis_right = axis_tmp.twinx()
    axis_right.set_ylabel('bin size = ' + str(psth_dt_ms) + 'ms')
    axis_right.tick_params(which='both', right=False, labelright=False)

    # share x axis (time) among all horizontal plots
    [axes[0].get_shared_x_axes().join(axes[0], axes[n + 1]) for n in range(1)]
    # disable x tick labels for spike raster plot
    axes[0].set_xticklabels([])

    # other settings
    axes[0].set_xlim(0, t_max)
    axes[1].set_xlabel('t [ms]')
    axes[1].legend(prop={'size': 8})

    return fig, axes, t_values, psth, psth_smooth


def plot_spikes_psth_comparison(spiketimes_mod_ms, info, idx_pop_oi, spiketimes_rec_ms, t_values_mod, t_values_rec,
                                psths_mod=None, psths_rec=None, psths_mod_smooth=None, psths_rec_smooth=None,
                                dot_size=2, recording_offset_ms=0, t_const_event_marker=None,
                                t_const_span_marker=None, b_black_background=False):
    """Plot spiketimes of all trials of a recorded neuron (spiketimes_rec_ms) into one subplot and spiketimes of all
    model neurons from one population (idx_pop_oi) into a second subplot. Plot peri-stimulus time histograms for both
    in a separate subplot. One figure per spikemon (i.e. model run)

    :param spiketimes_mod_ms: list of lists of spike times of the model neurons (one sublist per neuron or trial)
    :type spiketimes_mod_ms: [[float]]
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param idx_pop_oi: index of the model neuron population whose neurons are to be plotted
    :type idx_pop_oi: int
    :param spiketimes_rec_ms: list of lists of spike times of the recorded neurons (one sublist per neuron or trial)
    :type spiketimes_rec_ms: [[float]]
    :param t_values_mod: (list of) list(s) of time values of psth bins (left edges) in milliseconds for model neurons
    :type t_values_mod: [float] or [[float]]
    :param t_values_rec: (list of) list(s) of time values of psth bins (left edges) in milliseconds for recorded neurons
    :type t_values_rec: [float] or [[float]]
    :param psths_mod: [default=None] (list of) psth(s) for model neurons, as returned by p_analysis.get_psth(). pass
        either psths_mod, psths_mod_smoothed or both.
    :type psths_mod: [float] or [[float]] or None
    :param psths_rec: [default=None] (list of) psth(s) for recorded neurons, as returned by p_analysis.get_psth(). pass
        either psths_rec, psths_rec_smoothed or both.
    :type psths_rec: [float] or [[float]] or None
    :param psths_mod_smooth: [default=None] smoothed psth(s) for model neurons as returned by p_analysis.get_psth()
    :type psths_mod_smooth: [float] or [[float]] or None
    :param psths_rec_smooth: [default=None] smoothed psth(s) for recorded neurons as returned by p_analysis.get_psth()
    :type psths_rec_smooth: [float] or [[float]] or None
    :param dot_size: [default=2] size for spike markers in dot raster plot
    :type dot_size: int or float
    :param recording_offset_ms: [default=0] plots of recording data will start at this timepoint into the recording [ms]
    :type recording_offset_ms: int or float
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :return:
        - fig: figure handle(s)
        - axes: list of list of axis handles
    :rtype:
        - fig: [matplotlib.figure.Figure]
        - axes: [[matplotlib.axes._subplots.AxesSubplot]]
    """

    # if info, t_values, psths and/or psths_smooth are a single object/list, convert them to a list of a objects/lists
    if type(info) is not list:
        info = [info]
    if type(t_values_mod[0]) is not list:
        t_values_mod = [t_values_mod]
    if type(t_values_rec[0]) is not list:
        t_values_rec = [t_values_rec]
    if psths_mod is not None and type(psths_mod[0]) is not list:
        psths_mod = [psths_mod]
    if psths_rec is not None and type(psths_rec[0]) is not list:
        psths_rec = [psths_rec]
    if psths_mod_smooth is not None and type(psths_mod_smooth[0]) is not list:
        psths_mod_smooth = [psths_mod_smooth]
    if psths_rec_smooth is not None and type(psths_rec_smooth[0]) is not list:
        psths_rec_smooth = [psths_rec_smooth]

    # check inputs
    assert all(isinstance(item, list) for item in spiketimes_mod_ms), \
        "spiketimes_mod_ms must be a list of lists"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert type(idx_pop_oi) is int, "idx_pop_oi must be an integer"
    assert type(t_values_mod) is list and type(t_values_rec) is list, "t_values_mod/rec must be (list of) list(s)"
    assert type(psths_mod) is list or psths_mod is None, "psths_mod must be a (list of) list(s)"
    assert type(psths_rec) is list or psths_rec is None, "psths_rec must be a (list of) list(s)"
    assert type(psths_mod_smooth) is list or psths_mod_smooth is None, "psths_mod_smooth must be a (list of) list(s)"
    assert type(psths_rec_smooth) is list or psths_rec_smooth is None, "psths_mod_smooth must be a (list of) list(s)"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')
    fig = []
    axes = []
    n_figs = len(t_values_mod)
    n_gridcols = 1

    if n_figs > 20:
        print("WARNING: More than 20 figures are about to be created...")

    # loop through list of spike monitors
    for mon in range(n_figs):
        # create figure and set up grid for subplots
        n_populations = info[mon]['n_populations']
        n_gridrows = 3
        fig_tmp = plt.figure(figsize=(16, 9))
        fig.append(fig_tmp)
        plt.rcParams.update({'font.size': 12})
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
        axes.append([])
        color_map_light = get_color_list(n_populations, color_map_in=C_COLORS_LIGHT)
        color_map_dark = get_color_list(n_populations, color_map_in=C_COLORS)

        # plot all spikes for population of model neurons and recordings of real neuron
        for subplt in range(2):
            # set up subplot
            axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (subplt, 0), colspan=n_gridcols, rowspan=1)
            axes[mon].append(axis_tmp)
            axes[mon][subplt].set_ylabel(info[mon]['population_ids'][idx_pop_oi] + ' #')
            # add right hand axis for population size
            axis_right = axes[mon][subplt].twinx()
            if subplt == 0:
                axes[mon][subplt].set_title('Spike times of model neurons', weight='bold')
            else:
                axes[mon][subplt].set_title('Spike times of recorded neurons', weight='bold')
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axes[mon][subplt].axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            if t_const_span_marker is not None:
                axes[mon][subplt].axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
                if subplt == 0:
                    axes[mon][subplt].text(t_const_span_marker[0] + (t_const_span_marker[1] - t_const_span_marker[0])/2,
                                           1, "playback", color=C_COLORS_GREY[6], zorder=1, verticalalignment='bottom',
                                           horizontalalignment='center')
            if subplt == 0:
                # go through model neurons and plot all spikes for each neuron
                for nrn in range(len(spiketimes_mod_ms)):
                    axes[mon][subplt].plot(spiketimes_mod_ms[nrn], [nrn] * len(spiketimes_mod_ms[nrn]), '.',
                                           color=color_map_dark[idx_pop_oi], markersize=dot_size)
                # set y-axis limits and sample size for right axis
                axes[mon][subplt].set_ylim(-0.5, len(spiketimes_mod_ms) - 0.5)
                axis_right.set_ylabel('n = ' + str(len(spiketimes_mod_ms)) + '/' +
                                      str(info[mon]['population_sizes'][idx_pop_oi]))
            else:
                # go through list entries of spiketimes_rec_ms (neurons or trials) and plot all spikes for each
                spike_times_rec_offset = []
                for nrn in range(len(spiketimes_rec_ms)):
                    # subtract offset from time values
                    spike_times_rec_offset.append([spiketimes_rec_ms[nrn][idx]
                                                   for idx in range(len(spiketimes_rec_ms[nrn]))])
                    axes[mon][subplt].plot([spike_times_rec_offset[nrn][i] - recording_offset_ms
                                            for i in range(len(spike_times_rec_offset[nrn]))],
                                           [nrn] * len(spike_times_rec_offset[nrn]), '.',
                                           color='black', markersize=dot_size)
                # set y-axis limits and sample size for right axis
                axes[mon][subplt].set_ylim(-0.5, len(spiketimes_rec_ms) - 0.5)
                axis_right.set_ylabel('n = ' + str(len(spiketimes_rec_ms)))

            axis_right.tick_params(which='both', right=False, labelright=False)

        # offset recording t_values for plot
        t_values_rec_offset_cur = [t_values_rec[mon][i] - recording_offset_ms for i in range(len(t_values_rec[mon]))]

        # set up subplot for psths
        axis_tmp = plt.subplot2grid((n_gridrows, n_gridcols), (2, 0), colspan=n_gridcols, rowspan=1)
        axis_tmp.set_title('Peri-stimulus-time histogram', weight='bold')
        axis_tmp.set_ylabel('average spike rate [Hz]')
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
        if t_const_event_marker is not None:
            axis_tmp.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
        if t_const_span_marker is not None:
            axis_tmp.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
        # plot psths
        if psths_mod:
            axis_tmp.plot(t_values_mod[mon], psths_mod[mon], color=color_map_dark[idx_pop_oi], label='model psth')
        if psths_rec:
            axis_tmp.plot(t_values_rec_offset_cur, psths_rec[mon], color='black', label='recorded psth')
        if psths_mod_smooth:
            axis_tmp.plot(t_values_mod[mon], psths_mod_smooth[mon], '--', color=color_map_light[idx_pop_oi],
                          label='smoothed model psth')
        if psths_rec_smooth:
            axis_tmp.plot(t_values_rec_offset_cur, psths_rec_smooth[mon], '--', color=C_COLORS_GREY[4],
                          label='smoothed recorded psth')
        axes[mon].append(axis_tmp)
        # add right hand axis for bin size text
        axis_right = axis_tmp.twinx()
        axis_right.tick_params(which='both', right=False, labelright=False)

        # share x axis (time) among all horizontal plots
        [axes[mon][0].get_shared_x_axes().join(axes[mon][0], axes[mon][n + 1]) for n in range(2)]
        # disable x tick labels for spike raster plots and autoscale all axes
        [axes[mon][n].set_xticklabels([]) for n in range(2)]

        # other settings
        axes[mon][0].set_xlim(0, info[mon]['sim_time'])
        axes[mon][2].set_xlabel('t [ms]')
        axes[mon][2].legend(prop={'size': 8})

    return fig, axes


def plot_fi_curve(statemons, spikemons, config, info):
    """Plot frequency response curve for varying input current (F-I curve). Needs results of a simulation with range
    of amplitudes for input step current, i.e. multiple simulation runs with varying input amplitude.

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :return:
        - fig: figure handle
        - axis: axis handle
    :rtype:
        - fig: matplotlib.figure.Figure
        - axis: matplotlib.axes._subplots.AxesSubplot
    """

    # if statemons, spikemons and neuron_idx contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(spikemons) is not list:
        spikemons = [statemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.statemonitor.StateMonitor) for item in statemons) or \
        all(isinstance(item, SimpleNamespace) for item in statemons), \
        "statemons must be (list of) b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be (list of) b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(statemons) is len(spikemons) is len(info), \
        "statemons, spikemons and info must all have the same number of elements"

    # check if input current was recorded in statemons
    if not hasattr(statemons[0], 'Ie'):
        print("p_plot.plot_fi_curve(): 'Ie' is not an attribute of statemons[0], i.e. no input current was recorded" +
              " => fI curve is not being plotted")
        return None, None

    # create figure and set up grid for subplots
    fig = plt.figure(figsize=(16, 9))
    axis = fig.add_subplot(1, 1, 1)
    h_lines = []
    n_neurons = sum(info[0]['population_sizes'])
    n_runs = len(statemons)
    cmap = plt.cm.get_cmap(CMAP_NAME_CONTINUOUS)

    # loop through runs
    for mon in range(n_runs):
        freq_cur_run = []
        current_cur_nrn = []
        # loop through populations and calculate f-I curve
        for nrn in range(n_neurons):
            spikes_cur_nrn = np.array(spikemons[mon].t[spikemons[mon].i == nrn] / b2.ms)
            onset_stim_ms = config['input_current']['t_start'][0][0]
            offset_stim_ms = onset_stim_ms + config['input_current']['duration'][0][0]
            stim_duration = offset_stim_ms - onset_stim_ms
            b_spikes_during_stim = np.logical_and(spikes_cur_nrn >= onset_stim_ms, spikes_cur_nrn < offset_stim_ms)
            n_spikes_cur_nrn = sum(b_spikes_during_stim)
            freq_cur_run.append(n_spikes_cur_nrn / stim_duration * 1000)
            i_current_onset = np.nonzero(statemons[mon].t >= onset_stim_ms * b2.ms)[0][0]
            current_cur_nrn.append(statemons[mon].Ie[nrn, i_current_onset] / b2.nA)

        # plot voltage traces
        if info[mon]['free_parameter_values']:
            h_lines.append(axis.plot(current_cur_nrn, freq_cur_run, color=cmap(mon / n_runs),
                                     label='%.2f' % float(info[mon]['free_parameter_values'][0]))[0])
        else:
            h_lines.append(axis.plot(current_cur_nrn, freq_cur_run, color=cmap(mon / n_runs),
                                     label='_')[0])

    # set axis properties
    axis.set_xlabel('Input current [nA]')
    axis.set_ylabel('Average firing frequency [Hz]')

    # add legend
    if n_runs > 1:
        axis.legend(title=info[0]['free_parameter_name'], prop={'size': 8})

    return fig, axis


def plot_psp_psc(statemons, config, info, b_share_y_axis=False, t_const_event_marker=None, t_const_span_marker=None,
                 b_black_background=False, b_maxmin_marker=True, b_derivative=False, i_colors=None):
    """Plot voltage traces (post-synaptic potential) and synaptic currents (post-synaptic current) for all neurons.
    Works for simulations in which each neuron recieves a single spike at t=0 from a separate spikegenerator.
    on_pre equations must be excitatory first, inhibitory second (see usage of on_pre_idx_gen below).

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param b_share_y_axis: [default=True] if true, all voltage trace subplots share the same y-axis limits
    :type b_share_y_axis: bool
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param t_const_span_marker: [default=None] timespan (list of x_min and x_max) at which to plot a shaded rectangle
    :type t_const_span_marker: [int, int] or [float, float] or None
    :param b_black_background: [default=False] plot on black background
    :type b_black_background: bool
    :param b_maxmin_marker: [default=False] mark maximum and minimum of each trace (if above/below initial voltage)
    :type b_maxmin_marker: bool
    :param b_derivative: plot derivative of each trace, rescaled and shifted to trace plot
    :type b_derivative: bool
    :param i_colors: [default=None] list of indices to colors of the colormap (i.e. order of colors) of length n_colors
    :type i_colors: list or None
    :return:
        - fig: figure handle(s)
        - axes: axis handles, one sublist per figure
    :rtype:
        - fig: [matplotlib.figure.Figure]
        - axes: [[matplotlib.axes._subplots.AxesSubplot]]
    """

    # if statemons and spikemons contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.statemonitor.StateMonitor) for item in statemons) or \
        all(isinstance(item, SimpleNamespace) for item in statemons), \
        "statemons must be a list of b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"

    # set figure style
    if b_black_background:
        plt.style.use('seaborn-dark')
    fig = []
    axes = []
    n_figs = len(statemons)
    n_gridcols = 2
    fontsize = 9

    # loop through list of state monitors
    for mon in range(n_figs):
        print(". plotting PSPs and PSCs for all neurons")

        # create figure and set up grid for subplots
        n_neurons = statemons[mon].v.shape[0]
        n_gridrows = n_neurons
        fig_tmp = plt.figure(figsize=(10, 6))
        fig.append(fig_tmp)
        plt.rcParams.update({'font.size': 12})
        matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
        axes.append([])
        if i_colors is None:
            i_colors = list(range(n_neurons))
        color_map_light = get_color_list(n_neurons, color_map_in=C_COLORS_LIGHT, i_colors=i_colors)
        color_map_dark = get_color_list(n_neurons, color_map_in=C_COLORS, i_colors=i_colors)

        # loop through all neurons in current statemonitor and plot the voltage trace and synaptic current
        max_voltage = None
        min_voltage = None
        max_current = None
        min_current = None
        for nrn in range(n_neurons):
            # set up subplot for PSP plot
            axis = plt.subplot2grid((n_gridrows, n_gridcols), (nrn, 1), colspan=1, rowspan=1)
            axes[mon].append(axis)
            axis.set_ylabel('$\mathrm{V_m}$ [mV]')
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axis.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            if t_const_span_marker is not None:
                axis.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            # plot voltage traces
            if config['generator']['on_pre_idx_gen'][nrn] == 0:
                current_trace = statemons[mon].Ise[nrn, :] / b2.pA
                label = 'EPSP ' + info[mon]['population_ids'][nrn]
            elif config['generator']['on_pre_idx_gen'][nrn] == 1:
                current_trace = statemons[mon].Isi[nrn, :] / b2.pA
                label = 'IPSP ' + info[mon]['population_ids'][nrn]
            axis.plot(statemons[mon].t / b2.ms, statemons[mon].v[nrn, :] / b2.mV, color=color_map_dark[nrn],
                      label=label)
            # get max/min for y-axis
            max_voltage_cur = np.max(statemons[mon].v[nrn, :] / b2.mV)
            min_voltage_cur = np.min(statemons[mon].v[nrn, :] / b2.mV)
            if max_voltage is None or max_voltage_cur > max_voltage:
                max_voltage = max_voltage_cur
            if min_voltage is None or min_voltage_cur < min_voltage:
                min_voltage = min_voltage_cur
            # plot marker at maximum / minimum voltage
            if b_maxmin_marker:
                initial_voltage = statemons[mon].v[nrn, 0] / b2.mV
                if max_voltage_cur > initial_voltage:
                    i_max_v = np.argmax(statemons[mon].v[nrn, :] / b2.mV)
                    t_max_v = statemons[mon].t[i_max_v] / b2.ms
                    amp_peak = max_voltage_cur - initial_voltage
                    axis.plot(t_max_v, max_voltage_cur, 'xk')
                    axis.text(t_max_v+1, max_voltage_cur, 'time-to-peak: ' + str(round(t_max_v, 1)) + 'ms, amplitude: '
                              + str(round(amp_peak, 1)) + 'mV', verticalalignment='bottom', fontsize=fontsize)
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_max_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axis.plot(t_rise, v_25_percent, 'xk')
                    axis.text(t_rise+1, v_25_percent, '25% rise time: ' + str(round(t_rise, 1)) + 'ms',
                              verticalalignment='top', fontsize=fontsize)
                if min_voltage_cur < initial_voltage:
                    i_min_v = np.argmin(statemons[mon].v[nrn, :] / b2.mV)
                    t_min_v = statemons[mon].t[i_min_v] / b2.ms
                    amp_peak = min_voltage_cur - initial_voltage
                    axis.plot(t_min_v, min_voltage_cur, 'xk')
                    axis.text(t_min_v+1, min_voltage_cur, 'time-to-peak: ' + str(round(t_min_v, 1)) + 'ms, amplitude: '
                              + str(round(amp_peak, 1)) + 'mV', verticalalignment='top', fontsize=fontsize)
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(statemons[mon].v[nrn, :], i_min_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = statemons[mon].v[nrn, i_25_percent] / b2.mV
                    axis.plot(t_rise, v_25_percent, 'xk')
                    axis.text(t_rise+1, v_25_percent, '25% rise time: ' + str(round(t_rise, 1)) + 'ms',
                              verticalalignment='bottom', fontsize=fontsize)
            # plot derivative of trace
            if b_derivative:
                gradient = np.gradient(statemons[mon].v[nrn, :] / b2.mV)
                min_g = min(gradient)
                max_g = max(gradient)
                scaling_factor_g = 1 / (max_g - min_g) * (max_voltage_cur - min_voltage_cur)
                gradient_rescaled = gradient * scaling_factor_g
                gradient_shifted = gradient_rescaled - min_g * scaling_factor_g + min_voltage_cur
                axis.plot(statemons[mon].t / b2.ms, gradient_shifted, color=[.6, .6, .6], label='gradient')
                if b_maxmin_marker:
                    if abs(max_g) > abs(min_g):
                        t_max_g = statemons[mon].t[np.argmax(gradient)] / b2.ms
                        axis.plot(t_max_g, max(gradient_shifted), 'xk')
                        axis.text(t_max_g, max(gradient_shifted), '  ttp: ' + str(round(t_max_g, 2)) + 'ms',
                                  verticalalignment='bottom', color=[.5, .5, .5], fontsize=fontsize)
                    if abs(max_g) < abs(min_g):
                        t_min_g = statemons[mon].t[np.argmin(gradient)] / b2.ms
                        axis.plot(t_min_g, min(gradient_shifted), 'xk')
                        axis.text(t_min_g, min(gradient_shifted), '  peak: ' + str(round(t_min_g, 2)) + 'ms',
                                  verticalalignment='top', color=[.5, .5, .5], fontsize=fontsize)
            axis.set_ylim(min_voltage_cur - 0.2 * abs(amp_peak),
                          max_voltage_cur + 0.2 * abs(amp_peak))

            # set up subplot for PSC plot ###############################################################
            axis = plt.subplot2grid((n_gridrows, n_gridcols), (nrn, 0), colspan=1, rowspan=1)
            axes[mon].append(axis)
            axis.set_ylabel('$\mathrm{I_s}$ [pA]')
            # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that timepoint
            if t_const_event_marker is not None:
                axis.axvline(t_const_event_marker, color=C_COLORS_GREY[9], linewidth=1)
            if t_const_span_marker is not None:
                axis.axvspan(t_const_span_marker[0], t_const_span_marker[1], facecolor=C_COLORS_GREY[9])
            # plot synaptic currents
            if config['generator']['on_pre_idx_gen'][nrn] == 0:
                current_trace = statemons[mon].Ise[nrn, :] / b2.pA
                label = 'EPSC ' + info[mon]['population_ids'][nrn]
            elif config['generator']['on_pre_idx_gen'][nrn] == 1:
                current_trace = statemons[mon].Isi[nrn, :] / b2.pA
                label = 'IPSC ' + info[mon]['population_ids'][nrn]
            axis.plot(statemons[mon].t / b2.ms, current_trace, color=color_map_light[nrn], label=label)
            # get max/min for y-axis
            max_current_cur = np.max(current_trace)
            min_current_cur = np.min(current_trace)
            if max_current is None or max_current_cur > max_current:
                max_current = max_current_cur
            if min_current is None or min_current_cur < min_current:
                min_current = min_current_cur
            # plot marker at maximum / minimum current
            if b_maxmin_marker:
                initial_current = current_trace[0]
                if max_current_cur > initial_current:
                    i_max_v = np.argmax(current_trace)
                    t_max_v = statemons[mon].t[i_max_v] / b2.ms
                    amp_peak = max_current_cur - initial_current
                    axis.plot(t_max_v, max_current_cur, 'xk')
                    axis.text(t_max_v+1, max_current_cur, 'time-to-peak: ' + str(round(t_max_v, 1)) + 'ms, amplitude: '
                              + str(round(amp_peak, 1)) + 'pA', verticalalignment='bottom', fontsize=fontsize)
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(current_trace, i_max_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = current_trace[i_25_percent]
                    axis.plot(t_rise, v_25_percent, 'xk')
                    axis.text(t_rise+1, v_25_percent, '25% rise time: ' + str(round(t_rise, 1)) + 'ms',
                              verticalalignment='top', fontsize=fontsize)
                if min_current_cur < initial_current:
                    i_min_v = np.argmin(current_trace)
                    t_min_v = statemons[mon].t[i_min_v] / b2.ms
                    amp_peak = min_current_cur - initial_current
                    axis.plot(t_min_v, min_current_cur, 'xk')
                    axis.text(t_min_v+1, min_current_cur, 'time-to-peak: ' + str(round(t_min_v, 1)) + 'ms, amplitude: '
                              + str(round(amp_peak, 1)) + 'pA', verticalalignment='top', fontsize=fontsize)
                    # get and mark 25% rise time
                    t_rise, i_25_percent = p_analysis.get_rise_time(current_trace, i_min_v, 0, percent=25,
                                                                    sampling_frequency_khz=1 /
                                                                    (statemons[mon].t[1]-statemons[mon].t[0]) / b2.kHz)
                    v_25_percent = current_trace[i_25_percent]
                    axis.plot(t_rise, v_25_percent, 'xk')
                    axis.text(t_rise+1, v_25_percent, '25% rise time: ' + str(round(t_rise, 1)) + 'ms',
                              verticalalignment='bottom', fontsize=fontsize)
            # plot derivative of current
            if b_derivative:
                gradient = np.gradient(current_trace)
                min_g = min(gradient)
                max_g = max(gradient)
                scaling_factor_g = 1 / (max_g - min_g) * (max_current_cur - min_current_cur)
                gradient_rescaled = gradient * scaling_factor_g
                gradient_shifted = gradient_rescaled - min_g * scaling_factor_g + min_current_cur
                axis.plot(statemons[mon].t / b2.ms, gradient_shifted, color=[.6, .6, .6], label='gradient')
                if b_maxmin_marker:
                    if abs(max_g) > abs(min_g):
                        t_max_g = statemons[mon].t[np.argmax(gradient)] / b2.ms
                        axis.plot(t_max_g, max(gradient_shifted), 'xk')
                        axis.text(t_max_g, max(gradient_shifted), '  peak: ' + str(round(t_max_g, 2)) + 'ms',
                                  verticalalignment='bottom', color=[.5, .5, .5], fontsize=fontsize)
                    if abs(max_g) < abs(min_g):
                        t_min_g = statemons[mon].t[np.argmin(gradient)] / b2.ms
                        axis.plot(t_min_g, min(gradient_shifted), 'xk')
                        axis.text(t_min_g, min(gradient_shifted), '  peak: ' + str(round(t_min_g, 2)) + 'ms',
                                  verticalalignment='top', color=[.5, .5, .5], fontsize=fontsize)
            axis.set_ylim(min_current_cur - 0.2 * abs(amp_peak),
                          max_current_cur + 0.2 * abs(amp_peak))

        # share y axis among voltage trace plots and x axis (time) among all horizontal plots
        [axes[mon][n].get_shared_x_axes().join(axes[mon][n], axes[mon][n+1]) for n in range(len(axes[mon]) - 1)]
        if b_share_y_axis:
            [axes[mon][n].get_shared_y_axes().join(axes[mon][n], axes[mon][n+2]) for n in range(0, len(axes[mon])-2, 2)]
            [axes[mon][n].get_shared_y_axes().join(axes[mon][n], axes[mon][n+2]) for n in range(1, len(axes[mon])-2, 2)]
        # disable x tick labels for plots except for last plot and autoscale all axes
        [axes[mon][n].set_xticklabels([]) for n in range(len(axes[mon])-2)]
        # [axes[mon][n].autoscale() for n in range(len(axes[mon]))]

        # set y-limits
        if b_share_y_axis:
            [axes[mon][n].set_ylim(1.1 * min_voltage, 0.9 * max_voltage) for n in range(0, len(axes[mon]), 2)]
            [axes[mon][n].set_ylim(1.1 * min_current, 0.9 * max_current) for n in range(1, len(axes[mon]), 2)]

        # set y tick labels of spike plot to population ids, and other settings
        axes[mon][0].set_xlim(0, 50)
        axes[mon][-2].set_xlabel('t [ms]')
        axes[mon][-1].set_xlabel('t [ms]')
        [axes[mon][n].legend(bbox_to_anchor=(0, 0, 1, 0.6)) for n in range(len(axes[mon]))]

    return fig, axes


def plot_input_current(statemons, info, config, t_const_event_marker=None, marker_linestyle='--'):
    """Plot input current as generated by p_input_setup currents from a .json config file.

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param config: dictionary as loaded from .json configuration file
    :type config: dict
    :param t_const_event_marker: [default=None] timepoint at which to plot a vertical line as marker
    :type t_const_event_marker: int or float or None
    :param marker_linestyle: linestyle for event marker (default: '--' = dashed line)
    :type marker_linestyle: str
    """

    # if statemons and info contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(statemons) is len(info), \
        "statemons and info must have the same number of elements"

    # create figure and set up grid for subplots (if b_average_aggregate, only create one figure
    axes = []
    fig = plt.figure(figsize=(16, 9))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 9})
    n_pops = len(info[0]['population_sizes'])
    n_gridcols = np.round(n_pops / 2).astype(int)
    n_gridrows = 2
    matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    color_map_dark = get_color_list(n_pops, color_map_in=C_COLORS, i_colors=[5, 5, 6, 6])

    # loop through populations
    for pop in range(n_pops):
        axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (pop % 2, np.floor(pop/2).astype(int)),
                                     colspan=1, rowspan=1))
        # if time for an event marker or span was passed, plot a vertical line or shaded rectangle at that time
        if t_const_event_marker is not None:
            axes[pop].axvline(t_const_event_marker, color='k', linewidth=1, linestyle=marker_linestyle, zorder=2.5)
        # plot input current
        color_cur = color_map_dark[pop]
        axes[pop].plot(statemons[0].t / b2.ms - config['misc']['playback_start'], statemons[0].Ie[pop, :] / b2.pA,
                       color=color_cur, linewidth=1.5, label='')
        # other settings
        axes[pop].set_xlabel('t [ms]')
        axes[pop].set_ylabel('Ie [pA]')
        axes[pop].set_xlim((-100, 100))

    # share x axis (time) among all  plots
    [axes[n].get_shared_x_axes().join(axes[n], axes[n+1]) for n in range(n_pops-1)]
    [axes[n].get_shared_y_axes().join(axes[n], axes[n+2]) for n in (0, 1)]

    return fig, axes


def dot_plot(values, group_names, x_label, y_label, value_colors=None, marker_size=16, b_paired=False, b_mean=True,
             b_jitter=False, jitter_thresh=None, jitter_offset=0.1, h_ax=None):
    """Dot plot of one or more groups (x-axis).

    :param values: list of lists of values to be plotted. one sublist per group. first group is at x=0, second at x=1...
    :type values: list
    :param group_names: list of group names (strings) that are used as x-tick labels. in order of sublists in values
    :type group_names: list
    :param x_label: x-axis label
    :type x_label: str
    :param y_label: y-axis label
    :type y_label: str
    :param value_colors: (list of) list(s) of color values, one color for each value. either one sublist per group, or
        one list that gets duplicated for each group. in that case it must be at least as long as the largest group
    :type value_colors: list or None
    :param marker_size: (list of) sizes of dots, one int per value OR one int that will be repeated for all points
    :type marker_size: list or int
    :param b_paired: if True, values in sublists of values are considered paired, i.e. values[0][0] <-> values[1][0] ...
        lines are drawn that connect paried values between groups
    :type b_paired: bool
    :param b_mean: if True, the group means are plotted as horizontal lines
    :type b_mean: bool
    :param b_jitter: if True, nearby dots are jittered along the x-axis to avoid overlap
    :type b_jitter: bool
    :param jitter_thresh: threshold distance between two consecutive (sorted) y-values, below which dots are to be
        jittered. If None, the threshold will be automatically calculated depending on the data.
    :type jitter_thresh: float
    :param jitter_offset: amount by which each dot is offset on the x-axis
    :type jitter_offset: float
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - ax: axis handles, one sublist per figure
    :rtype:
        - fig: matplotlib.figure.Figure
        - ax: matplotlib.axes._subplots.AxesSubplot
    """

    # set up figure
    n_groups = len(values)
    if not h_ax:
        fig = plt.figure(figsize=(4 + 2 * n_groups, 8))
        plt.rcParams.update({'font.size': 15})
        ax = plt.axes()
    else:
        ax = h_ax

    # if value_colors is a single list of color values, duplicate this list for each group
    if value_colors and not isinstance(value_colors[0][0], list):
        value_colors = [value_colors] * n_groups

    # if marker_size contains a single integer, repeat this for all values
    if isinstance(marker_size, int):
        marker_size = [[marker_size] * len(v) for v in values]

    # determine jitter threshold
    if b_jitter and not jitter_thresh:
        jitter_thresh = abs(values[0][-1] - values[0][0]) / 25

    # loop through groups
    for g in range(n_groups):
        n_values_cur = len(values[g])
        # jitter nearby dots along the x-axis
        if b_jitter:
            x_values = jitter_dots(values[g], g, jitter_thresh=jitter_thresh, jitter_offset=jitter_offset)
        else:
            x_values = [g] * n_values_cur
        if b_paired and g < n_groups - 1:
            for v in range(n_values_cur):
                plt.plot((x_values[v], x_values[v] + 1), (values[g][v], values[g + 1][v]), color=[.3, .3, .3],
                         linewidth=2)
        if b_mean:
            cur_mean = np.mean(values[g])
            plt.plot((g - 0.25, g + 0.25), (cur_mean, cur_mean), color=[.6, .6, .6], linewidth=4)
        if value_colors:
            for v in range(n_values_cur):
                plt.plot(x_values[v], values[g][v], '.', color=value_colors[g][v], markersize=marker_size[g][v])
        else:
            plt.plot(x_values, values[g], '.k', markersize=marker_size[g][v])

    # set axis labels, limits, etc
    ax.set_xticks(range(0, n_groups))
    ax.set_xticklabels(group_names)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(-.5, n_groups - .5)

    if not h_ax:
        return fig, ax
    else:
        return


def histogram_plot(data, group_names=None, group_colors=None, x_label='', y_label='', binsize=10, h_ax=None):
    """Plot a histogram of one or more groups of values (data).

    :param data: list of list(s) of values, one sublist per group
    :type data: list
    :param group_names: list of strings of group names
    :type group_names: list
    :param group_colors: list of strings of group names
    :type group_colors: list
    :param x_label: x-axis label
    :type x_label: str
    :param y_label: y-axis label
    :type y_label: str
    :param binsize: size of the histogram bins in the unit of the data
    :type binsize: int or float
    :param h_ax: handle to the axis in which the psths are to be plotted. if None (default) create new figure. if a
        handle is passed, this function does not have return values.
    :type h_ax: matplotlib.axes._subplots.AxesSubplot
    :return:
        - fig: figure handle
        - ax: axis handles, one sublist per figure
    :rtype:
        - fig: matplotlib.figure.Figure
        - ax: matplotlib.axes._subplots.AxesSubplot
    """

    # set up figure
    n_groups = len(data)
    if not h_ax:
        fig = plt.figure(figsize=(12, 7))
        plt.rcParams.update({'font.size': 15})
        ax = plt.axes()
    else:
        ax = h_ax

    # if value_colors is a single list of color values, duplicate this list for each group
    if group_colors and not isinstance(group_colors[0], list):
        group_colors = [group_colors] * n_groups

    # plot histograms
    bins_n = []
    max_data = np.max([np.max(d) for d in data])
    min_data = np.min([np.min(d) for d in data])
    bin_min = np.floor(min_data / binsize) * binsize
    bin_max = np.ceil(max_data / binsize) * binsize
    for g in range(n_groups):
        ax.axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
        if group_colors:
            n, _, _ = ax.hist(data[g], bins=np.arange(bin_min, bin_max + binsize, binsize), alpha=0.6,
                              label=group_names[g], color=group_colors[g])
        else:
            n, _, _ = ax.hist(data[g], bins=np.arange(bin_min, bin_max + binsize, binsize), alpha=0.6,
                              label=group_names[g])
        bins_n.append(n)
    max_bin_height = np.max([np.max(bins) for bins in bins_n])
    ax.set_xlim((bin_min, bin_max))
    ax.set_ylim((0, max_bin_height))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()

    return fig, ax


def jitter_dots(y_values, x_value, jitter_thresh, jitter_offset=0.1):
    """Jitter nearby dots (e.g. for a dot plot) along the x-axis to avoid overlap

    :param y_values: list of values that determine closeness (e.g. y-axis values)
    :type y_values: list
    :param x_value: original x-axis value, around which dots will be jittered
    :type x_value: int or float
    :param jitter_thresh: threshold distance between two consecutive (sorted) y-values, below which dots are to be
        jittered.
    :type jitter_thresh: float
    :param jitter_offset: amount by which each dot is offset on the x-axis
    :type jitter_offset: float
    :return: x_values: list of one x-value for each y-value. either the original value, or one slightly offset
    :rtype: x_values: list
    """

    # sort y-axis values
    sorted_values = np.sort(y_values)
    sort_idx = np.argsort(y_values)

    # initialize array for x-axis offsets for each point (all start at 0)
    offsets = np.zeros(len(sorted_values))
    # initialize variable to keep track of which offset to use next
    next_offset = jitter_offset
    # the sign of the offset alternates for consecutive close y-values
    sign_of_next_offset = 1
    # this keeps track of the direction of the first of each sets of offset, because initial offsets sometimes alternate
    sign_of_initial_offset = 1
    # this keeps track of the index of the last y-value without offset
    i_last_zero_offset_dot = 0

    # go through sorted y-values
    for d in range(1, len(sorted_values)):
        # calculate distance to last point without offset
        dist_to_prev = sorted_values[d] - sorted_values[i_last_zero_offset_dot]
        if dist_to_prev < jitter_thresh:
            # if this is the first of a set of offsets, save the sign of the initial offset
            if i_last_zero_offset_dot == d - 1:
                sign_of_initial_offset = sign_of_next_offset
            # set the offset of the current point
            offsets[sort_idx[d]] = next_offset * sign_of_next_offset
            # if this offset was negative switch sign for next offset
            if sign_of_next_offset == -1:
                sign_of_next_offset = 1
                if sign_of_initial_offset == 1:
                    next_offset = next_offset + jitter_offset
            else:  # otherwise switch sign and increase next offset
                sign_of_next_offset = -1
                if sign_of_initial_offset == -1:
                    next_offset = next_offset + jitter_offset
        else:  # if distance is not smaller than threshold do not offset
            # reset value and sign for next offset
            next_offset = jitter_offset
            # update index for last y-value without offset (i.e. this one)
            i_last_zero_offset_dot = d

    x_values = [x_value + offset for offset in offsets]

    return x_values


def arrow(ax_handle, base_xy, tip_xy, margin=0.25, width=0.13, head_length=0.25, head_width_factor=2.0,
          color=(.85, .85, .85), bidirectional=False, label=None, clip_on=True):
    """Plot an arrow in axis ax_handle.

    :param ax_handle: Handle of the axis in which to plot the arrow
    :type ax_handle: matplotlib.axes._subplots.AxesSubplot
    :param base_xy: 2-element list/tuple of x- and y-coordinates of the arrow base
    :type base_xy: list or tuple
    :param tip_xy: 2-element list/tuple of x- and y-coordinates of the arrow tip
    :type tip_xy: list or tuple
    :param margin: shorten the arrow on both sides by margin
    :type margin: float
    :param width: width of the arrow tail (passed to plt.arrow())
    :type width: float
    :param head_length: length of the arrow head (passed to plt.arrow())
    :type head_length: float
    :param head_width_factor: arrow head width will be width * head_width_factor
    :type head_width_factor: float
    :param color: arrow color (passed to plt.arrow() as facecolor and edgecolor)
    :type color: list or str
    :param bidirectional: if true, arrow will have heads on both sides
    :type bidirectional: bool
    :param label: label for the arrow which will be displayed in the legend
    :type label: str
    :param clip_on: if False, portions of the arrow outside of the axis will be visible
    :type clip_on: bool
    :return:    - midpoint_xy: x- and y-coordinates of the arow midpoint (can be used e.g. for text annotations)
                - angle_deg: angle of the arrow in degrees
                - arrow_handle: handle of the arrow object (a FancyArrow object)
    :rtype:     - midpoint_xy: tuple
                - angle_deg: numpy.ndarray
                - arrow_handle: matplotlib.patches.FancyArrow
    """

    arrow_len = pow(pow(tip_xy[0] - base_xy[0], 2) + pow(tip_xy[1] - base_xy[1], 2), 0.5)
    sign_xy = [np.sign(tip_xy[0] - base_xy[0]), np.sign(tip_xy[1] - base_xy[1])]
    angle_rad = np.arctan2(base_xy[1] - tip_xy[1], base_xy[0] - tip_xy[0])  # pre_y - post_y, pre_x - post_x
    angle_deg = np.rad2deg(angle_rad)
    cos_angle_abs = np.abs(np.cos(angle_rad))
    sin_angle_abs = np.abs(np.sin(angle_rad))
    arrow_len_xy = np.array([arrow_len * np.cos(angle_rad), arrow_len * np.sin(angle_rad)])
    arrow_len_abs_xy = np.abs(arrow_len_xy)
    arrow_len_xy_shrink = np.array([sign_xy[0] * (arrow_len_abs_xy[0] - 2 * margin * cos_angle_abs),
                                    sign_xy[1] * (arrow_len_abs_xy[1] - 2 * margin * sin_angle_abs)])
    arrow_start_xy = [base_xy[0] + sign_xy[0] * margin * cos_angle_abs,
                      base_xy[1] + sign_xy[1] * margin * sin_angle_abs]
    if bidirectional:
        midpoint_xy = ((base_xy[0] + tip_xy[0]) / 2, (base_xy[1] + tip_xy[1]) / 2)
        arrow_start_xy = midpoint_xy
        arrow_len_xy_shrink = arrow_len_xy_shrink / 2
    else:
        midpoint_xy = ((base_xy[0] + tip_xy[0]) / 2 - sign_xy[0] * head_length / 2 * cos_angle_abs,
                       (base_xy[1] + tip_xy[1]) / 2 - sign_xy[1] * head_length / 2 * sin_angle_abs)
    arrow_handle = ax_handle.arrow(arrow_start_xy[0], arrow_start_xy[1], arrow_len_xy_shrink[0], arrow_len_xy_shrink[1],
                                   width=width, head_length=head_length, head_width=head_width_factor * width,
                                   length_includes_head=True, fc=color, ec=color, label=label, clip_on=clip_on)
    if bidirectional:
        ax_handle.arrow(arrow_start_xy[0], arrow_start_xy[1], -arrow_len_xy_shrink[0], -arrow_len_xy_shrink[1],
                        width=width, head_length=head_length, head_width=head_width_factor * width,
                        length_includes_head=True, fc=color, ec=color, clip_on=clip_on)

    return midpoint_xy, angle_deg, arrow_handle


def abline(slope, intercept, color='k', ax=None, label=None, zorder=None):
    """Plot a line from slope and intercept"""

    if ax is None:
        ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    handle = ax.plot(x_vals, y_vals, '--', color=color, label=label, zorder=zorder)

    return handle
