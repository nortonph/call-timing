"""Functions generating input for brian2 model neurons: spike times for SpikeGenerator and step and ramp currents.
(generate_current partly reuses code from http://neuronaldynamics.epfl.ch) [call-time-model]
"""

import os
import math
import logging
import numpy as np
import brian2 as b2
from fcn import p_io, p_util
from timeit import default_timer as timer


def get_offset_noise(config):
    """Get as many random numbers sampled from the standard normal distribution (using brian2.randn()) as needed for
    noise offsets during current generation.

    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :return: offset_noise: list of numpy arrays of random numbers for current generation, one array per population
    :rtype: offset_noise: [numpy.ndarray] or [[], [], ...]
    """

    # check inputs
    assert type(config) is dict, "p_input.get_random_offsets(): arg config must be dictionary (as loaded from .json)"

    n_populations = len(config['misc']['population_id'])
    if config['input_current'] and 'offset_range' in config['input_current'] \
            and any(config['input_current']['offset_range']):
        # generated the random numbers needed for each population
        offset_noise = []
        for pop in range(n_populations):
            # get number of currents for this population. if t_start is a free parameter, this is contained in a sublist
            if type(config['input_current']['t_start'][0][0]) is not list:
                n_currents = len(config['input_current']['t_start'][pop])
            else:
                n_currents = len(config['input_current']['t_start'][0][pop])
            if config['input_current']['offset_range'][pop] > 0:
                n_rand_offsets = config['misc']['n_neurons_per_pop'][pop] * n_currents
                offset_noise.append(b2.randn(n_rand_offsets))
            else:
                offset_noise.append([])

        return offset_noise
    else:
        return [[]] * n_populations


def get_white_noise(config):
    """Get as many random numbers sampled from the standard normal distribution (using brian2.randn()) as needed for
    white noise during current generation (within step/ramp random sample-to-sample variability)

    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :return: white_noise: list list(s) of numpy arrays of random numbers for current generation. one sublist per
        population, one array per neuron. list of Nones if no numbers were generated (one None per population)
    :rtype: white_noise: list
    """

    # check inputs
    assert type(config) is dict, "p_input.get_random_offsets(): arg config must be dictionary (as loaded from .json)"

    n_populations = len(config['misc']['population_id'])
    if config['input_current'] and 'wnoise_cv' in config['input_current'] \
            and any(config['input_current']['wnoise_cv']):
        # generated the random numbers needed for each population
        white_noise = []
        for pop in range(n_populations):
            # get number of samples needed
            n_samples = 1 + np.ceil(config['misc']['sim_time'] / config['input_current']['wnoise_dt']).astype(int) + 1
            white_noise.append([])
            if config['input_current']['wnoise_cv'][pop] > 0:
                for nrn in range(config['misc']['n_neurons_per_pop'][pop]):
                    # get a large, fixed number of random floats (exceeding simulation time), so that used
                    # random numbers are consistent when changing simulation time, but keeping the same seed.
                    tmp_lots_of_random_numbers = b2.randn(1000000)
                    white_noise[pop].append(tmp_lots_of_random_numbers[0:n_samples])
            else:
                white_noise[pop].append([])

        return white_noise
    else:
        return [None] * n_populations


def setup_currents(config, offset_noise, white_noise, log_name, sim_time_ms=None, b_gen_curr_per_pop=None,
                   I_input_prev=None):
    """Generate and return time series of currents to be used as input onto model neurons, based on specifications in
    config (e.g. loaded from .json config file).

    :param config: dictionary of model parameters as loaded from .json configuration file
    :type config: dict
    :param offset_noise: list of numpy arrays of random numbers used in randomly offsetting current amplitudes.
        one array per population. empty list instead of array where no numbers were generated (one [] per pop)
    :type offset_noise: list
    :param white_noise: list of list(s) of numpy arrays of random numbers used for additive noise in current generation.
        one sublist per population, one array per neuron. list of Nones if no numbers were generated (one None per pop)
    :type white_noise: list
    :param log_name: name of logger object (standard python logging) created earlier, e.g. through p_io.set_up_loggers()
    :type log_name: str
    :param sim_time_ms: [default=None] duration of simulation in milliseconds. if None: load from config file
    :type sim_time_ms: int or None
    :param b_gen_curr_per_pop: [default=None] list of one bool per population. if true, generate currents only for the
        respective population and overwrite I_input_prev for those. if None, generate currents for all populations.
    :type b_gen_curr_per_pop: list or None
    :param I_input_prev: [default=None] output of this function from the previous run. Supply if b_gen_curr_per_pop.
    :type I_input_prev: brian2.TimedArray or None
    :return:
        - I_input: brian2.TimedArray containing all generated currents. (May be used in ode as Ie)
        - log_message: string containing message to be printed/logged
    :rtype:
        - I_input: brian2.input.timedarray.TimedArray
        - log_message: string
    """

    # check inputs
    assert type(config) is dict, "p_input.setup_currents(): argument config must be dictionary (as loaded from .json)"
    assert sim_time_ms is None or (isinstance(sim_time_ms, int) and sim_time_ms >= 0), \
        "argument 'sim_time_ms' must be an integer >= 0 or None"
    if b_gen_curr_per_pop:
        assert I_input_prev and type(b_gen_curr_per_pop) is list and len(b_gen_curr_per_pop) is \
            len(config['misc']['population_id']) and type(I_input_prev) == b2.input.timedarray.TimedArray, \
            "if b_gen_curr_per_pop (list of len n_pops), also supply I_input_prev (b2.input.timedarray.TimedArray)."
    if I_input_prev:
        assert b_gen_curr_per_pop, "if I_input_prev is passed, b_gen_curr_per_pop must also be passed (see docstring)"

    # misc parameters
    population_id = config['misc']['population_id']
    n_populations = len(population_id)
    n_per_pop = config['misc']['n_neurons_per_pop']
    if sim_time_ms is None:
        sim_time_ms = config['misc']['sim_time']

    # get logger object
    log = logging.getLogger(log_name)

    # get unit for input current
    if 'input_current' in config['parameter_units']:
        if hasattr(b2.units, config['parameter_units']['input_current']):
            unit = 'b2.' + config['parameter_units']['input_current']
        else:
            log.error("ERROR: Unrecognized unit '" + config['parameter_units']['input_current'] + "' for input_current")
            raise ValueError("Unrecognized unit '" + config['parameter_units']['input_current'] + "' for input_current")
    else:
        unit = '1'

    # generate input currents (Ie)
    if config['input_current']:
        timer_input_start = timer()
        # check if all input_currents are applied within simulation time
        for t in config['input_current']['t_start']:
            if any(values > sim_time_ms for values in t):
                log.warning("WARNING: input currents extend beyond simulation time (" + str(sim_time_ms) + "ms)!")
        input_current_t_start = config['input_current']['t_start']
        input_current_amp_start = config['input_current']['amp_start']
        input_current_amp_end = config['input_current']['amp_end']
        if 'power_ramp' in config['input_current']:
            b_power_ramp_all = config['input_current']['power_ramp']
        else:
            b_power_ramp_all = [None] * n_populations
        if 'offset_range' in config['input_current']:
            input_current_offset_range = config['input_current']['offset_range']
        else:
            input_current_offset_range = [None] * n_populations
        if 'wnoise_cv' in config['input_current']:
            input_current_wnoise_cv = config['input_current']['wnoise_cv']
            input_current_wnoise_dt = config['input_current']['wnoise_dt']
        else:
            input_current_wnoise_cv = [[]] * n_populations
            input_current_wnoise_dt = 1

        # generate input current for all populations (except for populations where b_gen_curr_per_pop is 0, if any)
        input_currents = []
        for pop in range(n_populations):
            if not b_gen_curr_per_pop or (b_gen_curr_per_pop and b_gen_curr_per_pop[pop]):
                input_currents.append(
                    generate_current(input_current_t_start[pop], sim_time_ms, b2.defaultclock.dt,
                                     [val * eval(unit) for val in input_current_amp_start[pop]],
                                     [val * eval(unit) for val in input_current_amp_end[pop]], n_neurons=n_per_pop[pop],
                                     offset_range=input_current_offset_range[pop] * eval(unit),
                                     offset_noise=offset_noise[pop],
                                     wnoise_cv=input_current_wnoise_cv[pop], wnoise_dt=input_current_wnoise_dt,
                                     white_noise=white_noise[pop], b_power_ramp=b_power_ramp_all[pop]))
            else:
                # generate zero currents (will be overwritten in timedarray by I_input_prev.values below)
                input_currents.append([[0] * n_per_pop[pop]] * b2.amp)
        if I_input_prev:
            max_len_curr_prev_run = max([len(I_input_prev.values[:, c]) for c in range(I_input_prev.values.shape[1])])
            I_input = timedarray_from_current(input_currents, unit_time=b2.defaultclock.dt, name='input_current',
                                              pad_to_length=max_len_curr_prev_run)
        else:
            I_input = timedarray_from_current(input_currents, unit_time=b2.defaultclock.dt, name='input_current')
        # if not all currents were re-generated, overwrite I_input from the previous run only with new values
        if b_gen_curr_per_pop:
            for pop in range(n_populations):
                if not b_gen_curr_per_pop[pop]:
                    idx_first_nrn_of_pop = p_util.get_abs_from_rel_nrn_idx(0, pop, n_per_pop)
                    idx_last_nrn_of_pop = idx_first_nrn_of_pop + n_per_pop[pop]
                    I_input.values[:, idx_first_nrn_of_pop:idx_last_nrn_of_pop] = \
                        I_input_prev.values[:, idx_first_nrn_of_pop:idx_last_nrn_of_pop]
        timer_input_end = timer()
        log_message = "Input currents generated in " + str(round(timer_input_end - timer_input_start, 1)) + " seconds."
    else:
        # generate zero current
        input_currents = []
        for pop in range(n_populations):
            input_currents.append(generate_current(0, sim_time_ms, b2.defaultclock.dt, 0 * eval(unit), 0 * eval(unit),
                                                   n_neurons=n_per_pop[pop]))
        I_input = timedarray_from_current(input_currents, unit_time=b2.defaultclock.dt, name='input_current')
        log_message = "As 'input_current' in config is empty, zero currents were generated for all populations."

    return I_input, log_message


def generate_current(t_start, sim_time_ms, unit_time, amplitude_start, amplitude_end=None, n_neurons=1,
                     offset_range=0*b2.nA, offset_noise=None, wnoise_cv=0, wnoise_dt=1, white_noise=None,
                     b_power_ramp=None):
    """Generate input current as numpy.array. t_start and amplitude_start must be ints/Quantities or lists with the same
    number of elements. Each element corresponds to one step current or ramp current in a sequence of currents.
    Elements must be ordered ascending in time. Returned array has n_neurons. Columns are identical if offset_range==0.
    
    :param t_start: integer or float or list of integers/floats with start times of current
    :type t_start: int or float or list
    :param sim_time_ms: duration of simulation in milliseconds
    :type sim_time_ms: int or float
    :param unit_time: brian2 unit of time step (dt between successive current values), e.g. 0.1*b2.ms. Ideally, to have
        the same time step as the simulation, set this to b2.defaultclock.dt. Can be set to longer, esp. without ramps.
    :type unit_time: b2.units.fundamentalunits.Quantity [unit=sec]
    :param amplitude_start: quantity or list of quantities (unit: ampere) of amplitudes at t_start.
    :type amplitude_start: b2.units.fundamentalunits.Quantity [unit=amp] or list
    :param amplitude_end: [default=None] quantity or list of quantities of amplitudes at t_end. Generated current
        ramps from amplitude_start at t_start to amplitude_end at t_end. If amplitude_end is None, [] or empty np.array,
        it is set to amplitude_start, i.e. only constant step currents are generated.
    :type amplitude_end: b2.units.fundamentalunits.Quantity [unit=amp] or list or None
    :param n_neurons: [default=1] number of neurons == number of columns in the returned current array
    :type n_neurons: int
    :param offset_range: [default=0] the given current amplitude gets offset by a random value within offset_range.
        Random values are normally distributed and can exceed these limits! (code for uniformly
        distributed random offsets commented out below)
    :type offset_range: b2.units.fundamentalunits.Quantity [unit=amp]
    :param offset_noise: [default=None] numpy array of random numbers used in randomly offsetting current amplitudes.
        Empty list: no offsets
    :type offset_noise: numpy.ndarray or list
    :param wnoise_cv: [default=0] coefficient of variation for white noise added to full current
    :type wnoise_cv: float
    :param wnoise_dt: time in milliseconds between successive random changes in input current according to white_noise
    :type wnoise_dt: float or int
    :param white_noise: [default=None] list of numpy arrays (n_neurons) of random numbers used for additive white noise
    :type white_noise: list
    :param b_power_ramp: [default=None] None or list of bools of len(t_start). where True or 1, ramp currents increase
        with power 2 instead of a linear ramp
    :type b_power_ramp: list or None
    :return: current: numpy.array to be used as input_current to model neurons. Convert to b2.TimedArray before use,
        using p_input.timedarray_from_current()
    :rtype: brian2.units.fundamentalunits.Quantity [unit=amp]
    """

    # if amplitude_end is None, set it to amplitude_start (i.e. only generate step currents)
    if amplitude_end is None or np.asarray(amplitude_end).size == 0:
        amplitude_end = amplitude_start

    # if t_start, amplitude_start and amplitude_end are single values of the correct type, convert to lists
    if type(t_start) is int:
        t_start = [t_start]
    if type(amplitude_start) is type(amplitude_end) is b2.units.fundamentalunits.Quantity:
        amplitude_start = [amplitude_start]
        amplitude_end = [amplitude_end]
    if b_power_ramp is not None and not isinstance(b_power_ramp, list):
        b_power_ramp = [b_power_ramp]

    # check inputs
    assert all(isinstance(item, int) for item in t_start) or all(isinstance(item, float) for item in t_start), \
        "t_start must be int or float or list of ints/floats"
    assert type(unit_time) is b2.units.fundamentalunits.Quantity, "unit_time must be a b2 Quantity with unit seconds"
    assert b2.units.fundamentalunits.have_same_dimensions(amplitude_start, b2.amp), \
        "amplitude_start must have unit of current, e.g. b2.namp"
    assert b2.units.fundamentalunits.have_same_dimensions(amplitude_end, b2.amp), \
        "amplitude_end must have unit of current, e.g. b2.namp"
    assert len(t_start) is len(amplitude_start) is len(amplitude_end), \
        "t_start, amplitude_start and (optional) amplitude_end must all have the same number of elements"
    assert all(t_start[i] <= t_start[i+1] for i in range(len(t_start) - 1)), \
        "t_start must be ordered ascending in time, (amplitude_start must be ordered correspondingly"

    # get simulation time step and calculate the number of time steps in one millisecond
    steps_per_ms = 1 / (unit_time / b2.ms)

    # initialize current with zeros
    current_size = 1 + np.round(sim_time_ms * steps_per_ms).astype(int) + 1  # +1 for t=0, +1 for trailing 0
    current = np.zeros((current_size, n_neurons)) * b2.amp

    # loop through passed elements
    for c in range(len(t_start)):
        # if amplitude is zero for both start and end: skip this loop iteration
        if amplitude_end[c] == amplitude_start[c] == 0:
            continue
        # calculate the first and last index to the current array corresponding to t_start
        i_start = np.round(t_start[c] * steps_per_ms).astype(int)
        # if another entry in t_start follow, then that is the end of this current. if not, the end of the simulation is
        if c < len(t_start) - 1:
            i_end = np.round(t_start[c+1] * steps_per_ms).astype(int)
        else:
            i_end = np.round(sim_time_ms * steps_per_ms).astype(int)
        n_steps = i_end - i_start
        if not amplitude_end[c] == amplitude_start[c]:
            # generate upward ramp current with different random offset for each neuron (but same for start and end)
            if offset_range and any([offset_noise[v] for v in range(len(offset_noise))]) and offset_range > 0:
                for nrn in range(n_neurons):
                    current_offset = offset_range * offset_noise[c*n_neurons + nrn] / 2
                    amp_start_randomized = amplitude_start[c] + current_offset
                    amp_end_randomized = amplitude_end[c] + current_offset
                    amp_diff = amp_end_randomized - amp_start_randomized
                    t = np.array(range(0, n_steps + 1))
                    if b_power_ramp is not None and b_power_ramp[c]:
                        ramp = np.array([math.pow(x/n_steps, 2) * amp_diff + amp_start_randomized for x in t]) * b2.amp
                    else:
                        slope = amp_diff / float(n_steps)
                        ramp = amp_start_randomized + t * slope
                    current[i_start: i_end + 1, nrn] = ramp
            else:
                amp_diff = amplitude_end[c] - amplitude_start[c]
                t = np.array(range(0, n_steps + 1))
                if b_power_ramp is not None and b_power_ramp[c]:
                    ramp = np.array([math.pow(x / n_steps, 2) * amp_diff + amplitude_start[c] for x in t]) * b2.amp
                else:
                    slope = amp_diff / float(n_steps)
                    ramp = amplitude_start[c] + t * slope
                for nrn in range(n_neurons):
                    current[i_start: i_end + 1, nrn] = ramp
        else:
            if offset_range and any([offset_noise[v] for v in range(len(offset_noise))]) and offset_range > 0:
                # generate step current with different random offset within offset_range for each neuron
                for nrn in range(n_neurons):
                    current_offset = offset_range * offset_noise[c*n_neurons + nrn] / 2
                    amp_start_randomized = amplitude_start[c] + current_offset
                    current[i_start: i_end + 1, nrn] = amp_start_randomized
                    # for uniform distribution: amplitude_start[c] + offset_range * b2.rand() - offset_range / 2
            else:
                # generate same step current for all neurons
                current[i_start: i_end + 1, :] = amplitude_start[c]

    # add white noise
    if wnoise_cv and white_noise and wnoise_cv > 0:
        for nrn in range(n_neurons):
            white_noise_full = np.concatenate(([white_noise[nrn][0]], np.repeat(white_noise[nrn][1:-1],
                                                                                int(wnoise_dt / (unit_time / b2.ms))),
                                               [white_noise[nrn][-1]]))
            current[:, nrn] = current[:, nrn] + current[:, nrn] * wnoise_cv * white_noise_full

    return current


def timedarray_from_current(currents, unit_time, name=None, pad_to_length=None):
    """Bundle multiple current arrays returned by p_input.generate_current() into a brian2.TimedArray.

    :param currents: (list of) numpy.array(s) returned by p_input.generate_current(), one for each neuron/-group
    :type currents: b2.units.fundamentalunits.Quantity [unit=amp] or list
    :param unit_time: brian2 unit of time step (dt between successive current values), e.g. 0.1*b2.ms. Ideally, to have
        the same time step as the simulation, set this to b2.defaultclock.dt (might not work otherwise).
    :type unit_time: b2.units.fundamentalunits.Quantity [unit=sec]
    :param name: [default: None] sets name parameter for returned brian2.TimedArray
    :type name: str or None
    :param pad_to_length: [default=None] currents will be padded with zeros to this length (i.e. number of time steps)
    :type pad_to_length: int or None
    :return: timed_array_current: brian2.TimedArray containing all passed currents
    :rtype: brian2.input.timedarray.TimedArray
    """

    # if currents is a single value of the correct type, convert it to a list
    if type(currents) is b2.units.fundamentalunits.Quantity:
        currents = [currents]

    # check inputs
    assert type(unit_time) is b2.units.fundamentalunits.Quantity, "unit_time must be a b2 Quantity with unit seconds"
    assert [b2.units.fundamentalunits.have_same_dimensions(currents[c], b2.amp) for c in range(len(currents))], \
        "currents must have unit of current, e.g. b2.namp"
    assert type(name) is str or name is None, "name must be a string or None"
    assert type(pad_to_length) is int or pad_to_length is None, "pad_to_length must be an int or None"

    # determine maximum length of currents for padding, unless pad_to_length was passed
    if pad_to_length:
        max_length = pad_to_length
    else:
        max_length = np.max([len(currents[c]) for c in range(len(currents))])

    # construct numpy.array to hold all currents from passed list
    n_current_columns = sum([currents[c].shape[1] for c in range(len(currents))])
    numpy_array_current = b2.zeros((max_length, n_current_columns))

    # loop through currents and pad with zero, so that all have the same length
    next_col = 0
    for c in range(len(currents)):
        last_col = next_col+currents[c].shape[1]
        numpy_array_current[:, next_col:last_col] = np.pad(currents[c], ((0, max_length-len(currents[c])), (0, 0)),
                                                           'constant')
        next_col += currents[c].shape[1]

    # add unit
    numpy_array_current = numpy_array_current * b2.amp

    # convert to b2.TimedArray and return
    timed_array_current = b2.TimedArray(numpy_array_current, dt=unit_time, name=name)
    return timed_array_current


def load_spiketimes_for_gen(path_to_spiketimes, neuron_nrs=None, trace_nrs=None, population_nrs=None,
                            recording_offsets_ms=None, b_single_population=False):
    """Load spiketimes from recordings of specific neurons to use as input for SpikegeneratorGroups.

    :param path_to_spiketimes: path to directory containing spiketimes pickle files (.pkl)
    :type path_to_spiketimes: str
    :param neuron_nrs: [default=None] list of integer indices to neurons from which spiketimes should be loaded.
        1-based indexing of alphanumerically sorted .pkl filenames in path_to_spiketimes. None: load from all files
    :type neuron_nrs: [int] or None
    :param trace_nrs: [default=None] list of list(s) of integer indices to traces of the respective neuron (one sublist
        per neuron) from which spiketimes should be loaded. 1-based indexing. None: load from all traces
    :type trace_nrs: [[int]] or None
    :param population_nrs: [default=None] list of integer identifiers for grouping neurons into populations (used for
        spikegenerator populations), one integer per neuron. Must be in ascending order. None: either all neurons belong
        to the same population, or each neuron is part of its own population, depending on b_single_population
    :type population_nrs: [int] or None
    :param recording_offsets_ms: [default=None] list of offsets in milliseconds, which will be added to spiketimes to
        account for differential alignment. one offset per neuron. if None, offset will be zero for all neurons
    :type recording_offsets_ms: [float] or None
    :param b_single_population: [default=False] if True, all neurons will be put in a single sublist, i.e. will belong
        to a single spikegenerator population when passed to run_simulation(). Otherwise each .pkl file will (e.g.
        recorded neuron) will make up a seperate sublist / spikegenerator population. If population_nrs is not None,
        this parameter is ignored
    :type b_single_population: bool
    :return: input_spiketimes_ms: list of list(s) of list(s) of spike times in milliseconds. one sublist per
        generator population (corresponds to model neuron population), one subsublist per trace, i.e. generator
        neuron. To be passed to run_simulation() for creation of spikegenerator groups
    :rtype: [[[float]]]
    """

    # get names of all .pkl files in the supplied path and sort them alphanumerically
    spiketimes_filenames = [f for f in os.listdir(path_to_spiketimes)
                            if os.path.isfile(os.path.join(path_to_spiketimes, f))
                            and f.endswith('.pkl')]
    spiketimes_filenames.sort()

    # if no neuron numbers were passed, use all .pkl files found in directory
    if neuron_nrs is None:
        neuron_nrs = [i+1 for i in range(len(spiketimes_filenames))]

    # if no recording_offsets were passed, set all to zero
    if recording_offsets_ms is None:
        recording_offsets_ms = [0] * len(neuron_nrs)

    # if no population_nrs were passed, use the same for all neurons, or one population per neuron
    if population_nrs is None:
        if b_single_population:
            population_nrs = [1] * len(neuron_nrs)
        else:
            population_nrs = list(range(1, len(neuron_nrs) + 1))

    # check inputs
    assert type(neuron_nrs) is list and all(isinstance(item, int) and item > 0 for item in neuron_nrs), \
        "neuron_nrs must be a list of integer indices to neurons from which to load spiketimes (1-based indexing)."
    assert type(trace_nrs) is list and all(isinstance(item, list) and item[0] > 0 for item in trace_nrs), \
        "trace_nrs must be a list of lists of integer indices to traces, one sublist per trace (1-based indexing)."
    assert all(len(trace_nrs[sublist]) == len(set(trace_nrs[sublist])) for sublist in range(len(trace_nrs))), \
        "sublists of trace_nrs may not contain duplicates"
    assert type(population_nrs) is list and all(isinstance(item, int) and item > 0 for item in population_nrs), \
        "population_nrs must be a list of integer indices to populations for grouping neurons (1-based indexing)."
    assert population_nrs == sorted(population_nrs), "population_nrs must be in ascending order (e.g. [1, 1, 2, 3])"
    if trace_nrs is not None:
        assert len(neuron_nrs) is len(trace_nrs) is len(population_nrs) is len(recording_offsets_ms), \
            "neuron_nrs, trace_nrs, population_nrs and recording_offsets_ms must have the same number of elements"
    else:
        assert len(neuron_nrs) is len(population_nrs) is len(recording_offsets_ms), \
            "neuron_nrs, population_nrs and recording_offsets_ms must have the same number of elements"

    # loop through populations
    input_spiketimes_ms = []
    for pop in set(population_nrs):
        spiketimes_cur_pop = []
        i_cur_pop = [i for i in range(len(population_nrs)) if population_nrs[i] == pop]
        neuron_nrs_cur_pop = [neuron_nrs[i_cur_pop[i]] for i in range(len(i_cur_pop))]
        for neuron_nr_cur in neuron_nrs_cur_pop:
            spiketimes_cur_nrn = p_io.load_spiketimes_from_pkl(path_to_spiketimes +
                                                               spiketimes_filenames[neuron_nr_cur-1])
            # if trace_nrs is not None, delete all traces from input_spiketimes_ms not in current sublist of trace_nr
            if trace_nrs is not None:
                # loop through traces in reverse order following and delete those not in trace_nrs
                # (deleting in normal order would invalidate the following indices)
                for t in sorted(range(len(spiketimes_cur_nrn)), reverse=True):
                    if t + 1 not in trace_nrs[neuron_nrs.index(neuron_nr_cur)]:
                        del spiketimes_cur_nrn[t]
            rec_off_cur = recording_offsets_ms[neuron_nrs.index(neuron_nr_cur)]
            print("| loaded spiketimes of " + str(len(spiketimes_cur_nrn)) + " traces for population " + str(pop) +
                  " from " + spiketimes_filenames[neuron_nr_cur-1] + ": " + str(spiketimes_cur_nrn))
            if rec_off_cur > 0:
                spiketimes_cur_nrn = [[trc[spk] + rec_off_cur for spk in range(len(trc))] for trc in spiketimes_cur_nrn]
                print("\\ spiketimes offset by " + str(rec_off_cur) + " ms.")
            spiketimes_cur_pop = spiketimes_cur_pop + spiketimes_cur_nrn
        input_spiketimes_ms.append(spiketimes_cur_pop)

    return input_spiketimes_ms


def get_rates_for_poisson_group(config, log_name, sim_time_ms=None):
    """Generate and return brian2.TimedArray of spike rates for a Poisson Group, based on specifications in
    config (e.g. loaded from .json config file), namely config['poisson_group'].

    :param config: dictionary of model parameters as loaded from .json configuration file
    :type config: dict
    :param log_name: name of logger object (standard python logging) created earlier, e.g. through p_io.set_up_loggers()
    :type log_name: str
    :param sim_time_ms: [default=None] duration of simulation in milliseconds. if None: load from config file
    :type sim_time_ms: int or None
    :return: timed_array_poisson_rate: Timed Array of spike rates at each time point of the simulation in Hz
    :rtype: brian2.TimedArray
    """

    # check inputs
    assert type(config) is dict, "p_input.setup_currents(): argument config must be dictionary (as loaded from .json)"
    assert sim_time_ms is None or (isinstance(sim_time_ms, int) and sim_time_ms >= 0), \
        "argument 'sim_time_ms' must be an integer >= 0 or None"

    # misc parameters
    if sim_time_ms is None:
        sim_time_ms = config['misc']['sim_time']

    # get logger object
    log = logging.getLogger(log_name)

    # generate timed array of firing rates for a Poisson Group
    if config['poisson_group']:
        # check if all rates are applied within simulation time
        for t_rate_end in [config['poisson_group']['t_start'][c] + config['poisson_group']['duration'][c]
                              for c in range(len(config['poisson_group']['t_start']))]:
            if t_rate_end > sim_time_ms:
                log.warning("WARNING: firing rates for poisson group extend beyond simulation time (" +
                            str(sim_time_ms) + "ms)!")
        poisson_group_t_start = config['poisson_group']['t_start']
        poisson_group_duration = config['poisson_group']['duration']
        poisson_group_rate_start = config['poisson_group']['rate_start']
        poisson_group_rate_end = config['poisson_group']['rate_end']

        # if poisson_group_rate_end is None, set it to poisson_group_rate_start (i.e. only generate step rates)
        if poisson_group_rate_end is None or np.asarray(poisson_group_rate_end).size == 0:
            poisson_group_rate_end = poisson_group_rate_start

        # get simulation time step and calculate the number of time steps in one millisecond
        steps_per_ms = 1 / (b2.defaultclock.dt / b2.ms)

        # initialize rate with zeros
        last_rate_end = max([poisson_group_t_start[r] + poisson_group_duration[r]
                             for r in range(len(poisson_group_t_start))])
        rate_size = 1 + np.round(last_rate_end * steps_per_ms).astype(int) + 1  # +1 for t=0, +1 for trailing 0
        rate = np.zeros(rate_size) * b2.Hz

        # loop through passed elements
        for r in range(len(poisson_group_t_start)):
            if poisson_group_duration[r] <= 0:
                # if deltaT is zero, we return a zero rate
                continue
            # calculate the first & last index to the rate array corresponding to t_start & duration for this step/ramp
            i_start = np.round(poisson_group_t_start[r] * steps_per_ms).astype(int)
            i_end = np.round((poisson_group_t_start[r] + poisson_group_duration[r]) * steps_per_ms).astype(int)
            n_steps = i_end - i_start
            if not poisson_group_rate_end[r] == poisson_group_rate_start[r]:
                # generate upward ramping rate
                slope = (poisson_group_rate_end[r] - poisson_group_rate_start[r]) / float(n_steps)
                ramp = [poisson_group_rate_start[r] + t * slope for t in range(0, n_steps + 1)]
                rate[i_start: i_end + 1] = ramp * b2.Hz
            else:
                # generate step rate
                rate[i_start: i_end + 1] = poisson_group_rate_start[r] * b2.Hz

        # convert to b2.TimedArray and return
        timed_array_poisson_rate = b2.TimedArray(rate, dt=b2.defaultclock.dt, name='poisson_rate')
        return timed_array_poisson_rate
    else:
        return None
