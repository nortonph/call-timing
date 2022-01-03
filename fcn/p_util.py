"""Various utility functions. (call-time-model)
"""

import copy
import numpy as np
import brian2 as b2
from types import SimpleNamespace


def get_rel_from_abs_nrn_idx(idx_nrn_absolute, population_sizes):
    """Get the neuron index of a model neuron in its population from its absolute index. Example: the simulation
    consists of two population, two neurons each. The third neuron has the absolute index 2 (0-based indexing) and the
    relative index 0, as it is the first index of the second population.

    :param idx_nrn_absolute: absolute index to the neuron
    :type idx_nrn_absolute: int
    :param population_sizes: list of numbers of neurons in all populations of the simulation,
        i.e. [2, 2] in the above example
    :type population_sizes: [int]
    :return:
        - idx_nrn_relative: relative index to the neuron in its population
        - idx_pop_of_nrn: index to the population, the neuron of interest is in, ie. 1 in the above example (2nd pop)
    :rtype:
        - idx_nrn_relative: int
        - idx_pop_of_nrn: int
    """

    # initialize output to None => produce error if these don't get overwritten
    idx_nrn_relative = None
    idx_pop_of_nrn = None

    # if the index is negative (i.e. index relative to the end), convert it to positive
    if idx_nrn_absolute < 0:
        idx_nrn_absolute = sum(population_sizes) + idx_nrn_absolute

    # loop through populations
    for i_pop in range(len(population_sizes)):
        # if sum of neurons in all populations up to (incl.) the current one is larger than the absolute neuron idx ...
        if idx_nrn_absolute < sum(population_sizes[0:i_pop+1]):
            # ... then the neuron is in the current population => get population index
            idx_pop_of_nrn = i_pop
            #
            if i_pop > 0:
                idx_nrn_relative = idx_nrn_absolute - sum(population_sizes[0:i_pop])
            else:
                idx_nrn_relative = idx_nrn_absolute
            break

    if idx_nrn_relative is None or idx_pop_of_nrn is None:
        raise ValueError('p_util.get_rel_from_abs_nrn_idx() did not succeed in finding relative neuron index and/or ' +
                         'population for absolute neuron index ' + str(idx_nrn_absolute) + ' and population sizes ' +
                         str(population_sizes))

    return idx_nrn_relative, idx_pop_of_nrn


def get_abs_from_rel_nrn_idx(idx_nrn_relative, idx_pop_of_nrn, population_sizes):
    """Get the absolute neuron index of a model neuron in its population from its relative index in its population.
    Example: the simulation consists of two population, two neurons each. The third neuron has the absolute index 2
    (0-based indexing) and the relative index 0, as it is the first index of the second population.

    :param idx_nrn_relative: relative index to the neuron in its population
    :type idx_nrn_relative: int
    :param idx_pop_of_nrn: index to the population, the neuron of interest is in, ie. 1 in the above example (2nd pop)
    :type idx_pop_of_nrn: int
    :param population_sizes: list of numbers of neurons in all populations of the simulation,
        i.e. [2, 2] in the above example
    :type population_sizes: [int]
    :return: idx_nrn_absolute: absolute index to the neuron
    :rtype: int
    """

    # check inputs
    assert idx_pop_of_nrn < len(population_sizes), 'population ' + str(idx_pop_of_nrn) + ' is not in ' +\
        'population_sizes, which has ' + str(len(population_sizes)) + ' elements. (indices follow 0-based indexing!)'
    size_pop_of_nrn = population_sizes[idx_pop_of_nrn]
    if idx_nrn_relative + 1 > size_pop_of_nrn:
        raise ValueError('p_util.get_abs_from_rel_nrn_idx(): relative neuron index ' + str(idx_nrn_relative) +
                         ' lies outside of population ' + str(idx_pop_of_nrn) + ' with size ' + str(size_pop_of_nrn) +
                         ' (indices follow 0-based indexing!)')

    idx_nrn_absolute = idx_nrn_relative + sum(population_sizes[0:idx_pop_of_nrn])

    return idx_nrn_absolute


def get_traces_from_monitors(statemons, b_keep_unit=True):
    """Get list of arrays (one array per neuron) in the format of recorded traces from statemons (simulation output).

    :param statemons: (list of) brian2 StateMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type statemons: SimpleNamespace or dict or list
    :param b_keep_unit: if False, get rid of brian2 unit (i.e. divide by brian2.mV)
    :type b_keep_unit: bool or int
    :return:    - traces: list of list(s) of numpy ndarray(s) of voltage traces. one sublist per monitor, one array per
                    neuron. Either with brian2 unit preserved, or without unit in millivolts (depends on b_keep_unit)
                - sampling_frequency_kHz: sampling frequency of the ouput traces in kHz
    :rtype:     - traces: [numpy.ndarray]
                - sampling_frequency_kHz: float
    """

    # if statemons and spikemons contain a single object, convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]

    # get sampling rate of input data
    dt = statemons[0].t[1] - statemons[0].t[0]
    sampling_frequency_khz = 1 / dt / b2.kHz

    traces = []
    for mon in range(len(statemons)):
        traces.append([])
        n_neurons = np.shape(statemons[mon].v)[0]
        for nrn in range(n_neurons):
            if b_keep_unit:
                traces[mon].append(statemons[mon].v[nrn, :])
            else:
                traces[mon].append(statemons[mon].v[nrn, :] / b2.mV)

    return traces, sampling_frequency_khz


def get_spiketimes_from_monitor(spikemon, info, i_pops=None):
    """Extract spiketimes from output of simulation (data from single spikemon; either from sim or loaded from .pkl).
    Return as list of lists of spiketimes (in milliseconds; one sublist per neuron).

    :param spikemon: brian2 SpikeMonitor - like(!) SimpleNamespace from out/...pkl file or dict from b2...get_states()
    :type spikemon: SimpleNamespace or dict
    :param info: dictionary containing additional information about simulation (as created by run_simulation())
    :type info: dict
    :param i_pops: [default=None] index or list of indices of the model neuron population(s) whose spiketimes are to be
        returned. default: all populations
    :type i_pops: int or list
    :return: spiketimes_ms: list of lists of spike times (in milliseconds; one sublist per neuron, independent of source
        population)
    :rtype: [[float]]
    """

    if type(i_pops) is not list:
        i_pops = [i_pops]

    # check input
    assert type(spikemon) is SimpleNamespace or type(spikemon) is dict, "spikemon must be a dict from dict from" + \
        "b2...get_states() or (if loaded through p_io.load_monitors) a SimpleNamespace object"
    assert type(info) is dict, "info must be a dictionary"
    assert all(isinstance(item, int) for item in i_pops), "i_pops must be an integer or list of integers"

    # if i_pops is None, get indices of all populations
    if i_pops == [None]:
        i_pops = list(range(len(info['population_ids'])))

    # loop through populations of interest and add all spiketimes to output list
    spiketimes_ms = []
    for pop in i_pops:
        i_first_nrn_of_pop = sum(info['population_sizes'][0:pop])
        if type(spikemon) is SimpleNamespace:
            spiketimes_ms += [(spikemon.t[spikemon.i == nrn] / b2.ms).tolist() for nrn in
                              range(i_first_nrn_of_pop, i_first_nrn_of_pop + info['population_sizes'][pop])]
        else:
            spiketimes_ms += [(spikemon['t'][spikemon['i'] == nrn] / b2.ms).tolist() for nrn in
                              range(i_first_nrn_of_pop, i_first_nrn_of_pop + info['population_sizes'][pop])]

    return spiketimes_ms


def subset_neuron_from_monitors(statemons, spikemons, info, idx_nrn_abs):
    """Get membrane potential, spiketimes etc. of a specific model neuron out of brian2 Monitors and return new monitors
    containing only this neuron. Monitors will be returned as dictionaries and will probably have to be saved and loaded
    first using p_io.save_monitors & p_io.load_monitors before they can be used in some analysis/plotting functions.

    :param statemons: list of brian2 StateMonitor-like(!) SimpleNamespace from .pkl file
    :type statemons: list
    :param spikemons: list of brian2 SpikeMonitor-like(!) SimpleNamespace from .pkl file
    :type spikemons: list
    :param info: list of dicts containing additional information about simulation (as created by run_simulation())
    :type info: list
    :param idx_nrn_abs: absolute index to the neuron to be extracted
    :type idx_nrn_abs: int
    :return:    - statemons_out: list of dictionaries containining statemonitor data for only the single neuron.
                - spikemons_out: list of dictionaries containining spikemonitor data for only the single neuron.
                - info_out: list of dictionaries containing additional information about the simulation
    :rtype:     - statemons_out: list
                - spikemons_out: list
                - info_out: list
    """

    # todo: deepcopy the namespace instead of making a dictionary? -> wouldn't have to be saved and loaded then...

    statemons_out = []
    spikemons_out = []
    info_out = copy.deepcopy(info)
    for mon in range(len(statemons)):
        # if the index is negative (i.e. index relative to the end), convert it to positive
        if idx_nrn_abs < 0:
            idx_nrn_abs = sum(info[mon]['population_sizes']) + idx_nrn_abs
        idx_nrn_rel, idx_pop_of_nrn = get_rel_from_abs_nrn_idx(idx_nrn_abs, info[mon]['population_sizes'])
        # get states and spikes of the neuron of interest from the monitors
        statemons_out.append(dict())
        statemons_out[mon]['t'] = statemons[mon].t
        statemons_out[mon]['N'] = statemons[mon].N
        statemons_out[mon]['v'] = b2.array([[v] for v in statemons[mon].v[idx_nrn_abs]]) * b2.volt
        spikemons_out.append(dict())
        spikemons_out[mon]['t'] = spikemons[mon].t[spikemons[mon].i == idx_nrn_abs]
        spikemons_out[mon]['i'] = b2.array([0] * spikemons[mon].count[idx_nrn_abs])
        spikemons_out[mon]['count'] = b2.array([spikemons[mon].count[idx_nrn_abs]])
        spikemons_out[mon]['N'] = b2.array(sum(spikemons_out[mon]['count']), dtype='int32')
        # update info dict
        info_out[mon]['population_sizes'] = [0 for _ in range(len(info[mon]['population_sizes']))]
        info_out[mon]['population_sizes'][idx_pop_of_nrn] = 1

    return statemons_out, spikemons_out, info_out


def trim_monitors_to_time_span(time_span_ms, statemons=None, spikemons=None):
    """Takes (list of) State- and SpikeMonitors, removes all values (t, v, spiketimes) that lie outside of a supplied
    time span and returns the trimmed monitors.

    :param time_span_ms: start and end point of time span to be trimmed to in milliseconds
    :type time_span_ms: list or tuple
    :param statemons: (list of) brian2 StateMonitor-like(!) SimpleNamespace from .pkl file
    :type statemons: list or SimpleNamespace
    :param spikemons: (list of) brian2 SpikeMonitor-like(!) SimpleNamespace from .pkl file
    :type spikemons: list or SimpleNamespace
    :return:    - statemons_out: list of namespaces containining statemonitor data containing only data within time_span
                - spikemons_out: list of namespaces containining spikemonitor data containing only data within time_span
    :rtype:     - statemons_out: list
                - spikemons_out: list
    """

    # if statemons and spikemons contain a single object, convert them to lists
    if statemons is not None and type(statemons) is not list:
        statemons = [statemons]
    if spikemons is not None and type(spikemons) is not list:
        spikemons = [spikemons]

    # trim states
    if statemons:
        statemons_out = copy.deepcopy(statemons)
        for mon in range(len(statemons_out)):
            idx_t = np.where(np.logical_and(statemons_out[mon].t >= time_span_ms[0] * b2.ms,
                                            statemons_out[mon].t <= time_span_ms[1] * b2.ms))[0]
            statemons_out[mon].t = statemons_out[mon].t[idx_t]
            statemons_out[mon].v = np.array([statemons_out[mon].v[n, idx_t] for n in range(len(statemons_out[mon].v))])\
                * b2.volt
            if hasattr(statemons[mon], 'Ie'):
                statemons_out[mon].Ie = np.array([statemons_out[mon].Ie[n, idx_t]
                                                  for n in range(len(statemons_out[mon].Ie))])
    else:
        statemons_out = None

    # trim spikes
    if spikemons:
        spikemons_out = copy.deepcopy(spikemons)
        for mon in range(len(spikemons_out)):
            idx_t = np.where(np.logical_and(spikemons_out[mon].t >= time_span_ms[0] * b2.ms,
                                            spikemons_out[mon].t <= time_span_ms[1] * b2.ms))[0]
            spikemons_out[mon].t = spikemons_out[mon].t[idx_t]
            spikemons_out[mon].i = spikemons_out[mon].i[idx_t]
    else:
        spikemons_out = None

    return statemons_out, spikemons_out


def trim_traces_to_time_span(time_span_ms, traces, samp_freq_khz):
    """Takes (list of) traces, removes all values that lie outside of a supplied
    time span and returns the trimmed monitors.

    :param time_span_ms: start and end point of time span to be trimmed to in milliseconds
    :type time_span_ms: list or tuple
    :param traces: (list of) voltage traces (or any other time series)
    :type traces: list
    :param samp_freq_khz: sampling frequency in kilo-Hertz of the trace data
    :type samp_freq_khz: float or int
    :return:    - traces_out: list of traces containing only data within time_span
    :rtype:     - traces_out: list
    """

    # if traces is a single list, convert it to list of list
    if type(traces[0]) is not list and type(traces[0]) is not np.ndarray:
        traces = [traces]

    # trim traces
    traces_out = copy.deepcopy(traces)
    for trc in range(len(traces_out)):
        idx_t = np.arange(time_span_ms[0] * samp_freq_khz, time_span_ms[1] * samp_freq_khz, 1)
        traces_out[trc] = traces_out[trc][idx_t]

    return traces_out


def downsample_trace(traces, downsampling_factor):
    """Down-sample traces by returning traces with only every nth sample preserved, where n = downsampling_factor

    :param traces: (list of) voltage traces (or any other time series)
    :type traces: list
    :param downsampling_factor: factor by which to compress trace, e.g. 4 means returned trace contains every 4th sample
    :type downsampling_factor: int
    :return:    - traces_out: list of downsampled traces
    :rtype:     - traces_out: list
    """

    assert type(downsampling_factor) is int and downsampling_factor > 0, "downsampling_factor must be a positive int"
    # if traces is a single list, convert it to list of list
    if type(traces[0]) is not list and type(traces[0]) is not np.ndarray:
        traces = [traces]

    # trim traces
    traces_out = []
    for trc in range(len(traces)):
        traces_out.append(traces[trc][np.arange(0, len(traces[trc]), downsampling_factor)])

    return traces_out


def integer_linspace(start, end, step_size, b_include_end=True):
    """Get a list of evenly spaced values between start (included) and end. Values will be integers. b_include_end
    determines whether end will be included. If so, the final value in the list can be larger than end if end > start or
    smaller than end if end < start.

    :param start: First value of the returned sequence.
    :type start: int
    :param end: End of the sequence. If b_round_up == True, the final sequence value will be >= end, otherwise <= end.
    :type end: int or float
    :param step_size: Integer value of the difference between two consecutive sequence values.
    :type step_size: int
    :param b_include_end: [default=True] If True, the absolute of the final value of the sequence will be the next
        larger integer >= abs(end), that is divisible by step_size. If False, it will be the next integer <= abs(end),
        that is divisible by step_size. If end is an integer that is divisible by step_size, it will be the final value
        if b_include_end == True.
    :type b_include_end: bool
    :return: sequence
    :rtype: [int]
    """

    assert step_size > 0, "p_util.integer_linspace(): step_size has to be larger than zero"
    if start == end:
        print("WARNING: p_util.integer_linspace(): start == end => returned sequence will be [start] if " +
              "b_include_end==True or [] if b_include_end==False")

    # this flag determines whether np.linspace() should include the rounded end value. Will only be set to false if
    # end == end_rounded and argument b_include_end was set to False
    b_include_end_rounded = True

    # round end value to be divisible by step size, either up or down, depending on whether it should be in the sequence
    if b_include_end:
        if end < start:
            end_rounded = (step_size * np.floor((end - start) / step_size)).astype(int) + start
        else:
            end_rounded = (step_size * np.ceil((end - start)/step_size)).astype(int) + start
    else:
        if end < start:
            end_rounded = (step_size * np.ceil((end - start) / step_size)).astype(int) + start
        else:
            end_rounded = (step_size * np.floor((end - start) / step_size)).astype(int) + start
        # if end is divisible by stepsize (i.e. equal to end_rounded) and should not be included, set flag for linspace
        if end == end_rounded:
            b_include_end_rounded = False

    # generate sequence
    if b_include_end_rounded:
        n_values = (abs(end_rounded - start) / step_size + 1).astype(int)
        sequence = np.linspace(start, end_rounded, n_values, dtype=int, endpoint=True).tolist()
    else:
        n_values = (abs(end_rounded - start) / step_size).astype(int)
        sequence = np.linspace(start, end_rounded, n_values, dtype=int, endpoint=False).tolist()

    return sequence


def integrate_poisson_group_into_pops(spikemons, spikemon_poisson, info, config, pop_id_poisson=None):
    """Integrate poisson group into neuron populations for spike monitor data and info dict (synapse indices and pop id)

    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param spikemon_poisson: single(!) SimpleNamespace from file or dict from b2...get_states() w/ poisson group spikes
    :type spikemon_poisson: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :param pop_id_poisson: [default=None] ID (name) for poisson group population. Default: load fron config
    :type pop_id_poisson: str
    :return:
        - spikemons_integrated: spike monitor data with integrated poisson group spikes
        - info_integrated: info dict with integrated poisson group info
    :rtype:
        - spikemons_integrated: SimpleNamespace or dict or list
        - info_integrated: dict or list
    """

    # if spikemons and info contain a single object, convert them to lists
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(spikemon_poisson) is not list:
        spikemon_poisson = [spikemon_poisson]
    else:
        assert len(spikemon_poisson) == 1, \
            "p_util.integrate_poisson_group_into_pops(): only implemented for single poisson group"
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemons) or \
        all(isinstance(item, SimpleNamespace) for item in spikemons), \
        "spikemons must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespace objects"
    assert all(isinstance(item, b2.monitors.spikemonitor.SpikeMonitor) for item in spikemon_poisson) or \
        all(isinstance(item, SimpleNamespace) for item in spikemon_poisson), \
        "spikemon_poisson must be a list of b2.SpikeMonitor or (if loaded through p_io.load_monitors) SimpleNamespaces"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(spikemons) is len(info), \
        "spikemons and info must have the same number of elements"

    # misc parameters
    n_nrn_psn = config['poisson_group']['n_neurons']
    spikemons_integrated = copy.deepcopy(spikemons)
    info_integrated = copy.deepcopy(info)
    if not pop_id_poisson:
        pop_id_poisson = config['poisson_group']['population_id']

    # loop through runs
    for mon in range(len(spikemons_integrated)):
        # concatenate poisson spikes with regular population spikes
        spikemons_integrated[mon].i = spikemons_integrated[mon].i + n_nrn_psn
        spikemons_integrated[mon].i = np.concatenate((spikemon_poisson[0].i, spikemons_integrated[mon].i))
        spikemons_integrated[mon].t = np.concatenate((spikemon_poisson[0].t, spikemons_integrated[mon].t))
        spikemons_integrated[mon].count = np.concatenate((spikemon_poisson[0].count, spikemons_integrated[mon].count))
        spikemons_integrated[mon].N = spikemon_poisson[0].N + spikemons_integrated[mon].N

        # add poisson population to info
        syn_post_idx_psn = [v+1 for v in config['poisson_group']['syn_post_idx']]
        n_syn_psn = len(syn_post_idx_psn)
        conn_prob_psn = config['poisson_group']['connection_probability']
        info_integrated[mon]['n_populations'] += 1
        info_integrated[mon]['population_ids'] = [pop_id_poisson] + info_integrated[mon]['population_ids']
        info_integrated[mon]['population_sizes'] = [n_nrn_psn] + info_integrated[mon]['population_sizes']
        info_integrated[mon]['syn_pre_idx'] = [0] * n_syn_psn + [v+1 for v in info_integrated[mon]['syn_pre_idx']]
        info_integrated[mon]['syn_post_idx'] = syn_post_idx_psn + [v+1 for v in info_integrated[mon]['syn_post_idx']]
        info_integrated[mon]['syn_weight'] = config['parameters_psn']['w'] + info_integrated[mon]['syn_weight']
        info_integrated[mon]['syn_delay'] = config['parameters_psn']['delay'] + info_integrated[mon]['syn_delay']
        info_integrated[mon]['connection_probability'] = conn_prob_psn + info_integrated[mon]['connection_probability']

    return spikemons_integrated, info_integrated
