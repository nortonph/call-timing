"""Main function for running simulations of spiking network models using brian2, based on equations and parameters
pulled from .json config files. Main function: run_simulation(). [call-time-model]
"""

# Philipp Norton, 2019-2021

# import libraries
import os
import sys
import json
import copy
import logging
import itertools
import numpy as np
import brian2 as b2
import multiprocessing
from datetime import datetime
from timeit import default_timer as timer

# import my functions
from fcn import p_input, p_io, p_util

# constant "global" variables
STANDALONE_CODE_GEN = False  # The store/restore mechanism is not supported in the C++ standalone!


def run_simulation(model_name, config_preloaded=None, log_name=None, run_id=0, sim_time_ms=None, clock_dt_ms=0.02,
                   input_spiketimes_ms=None, filename_out_arg=None, recorded_variables='v', recording_start_ms=0,
                   b_always_save_output=True, b_verbose=False):
    """Run a simulation using this model with the parameters supplied by a .json config file.

    :param model_name: filename of parameter config file (ending .json can be omitted from string)
    :type model_name: str
    :param config_preloaded: if a config dictionary is supplied here, this will be used to run the simulation, instead
        of loading a config from the .json file indicated by model_name
    :type config_preloaded: None or dict
    :param log_name: [default=None] name of logger object (standard python logging). if None, a logger will be created
        for console output only
    :type log_name: str or None
    :param run_id: [default=0] integer id used to identify this run with corresponding log files
    :type run_id: int
    :param sim_time_ms: [default=None] duration of simulation in milliseconds. if None: load from config file
    :type sim_time_ms: int or None
    :param clock_dt_ms: [default=0.02] simulation time step in milliseconds, between 0 and (including) 1
    :type clock_dt_ms: float
    :param input_spiketimes_ms: [default=None] list of list(s) of list(s) of spiketimes to be used as input onto model
        neurons using SpikeGeneratorGroups. Each sublist corresponds to a population that acts as input on the
        respective model neuron population if n_generators == n_populations. If n_generators > 1 and n_populations == 1,
        then all generators connect to the population. Each subsublist contains all spiketimes of one generator neuron.
        Condition for connections from generators to neurons can be set in config['generator']['condition']
    :type input_spiketimes_ms: [[[float]]] or None
    :param filename_out_arg: [default=None] optional filename for output .pkl file. if None, this is handled by
        p_io.load_config()
    :type filename_out_arg: None or str
    :param recorded_variables: (tuple of) name(s) of the variable(s) to be recorded in StateMonitor and saved to .pkl
    :type recorded_variables: str or tuple
    :param recording_start_ms: [default=0] start recording spikes and states at this timepoint into the simulation
    :type recording_start_ms: float or int
    :param b_always_save_output: if False, run is_worth_saving to check a certain condition for saving output to .pkl
    :type b_always_save_output: bool
    :param b_verbose: [default=False] if True, additional information is output during the run
    :type b_verbose: bool or int
    :return:  - filename_out: filename for output .pkl file
              - conditions_not_met: list of indices of conditions not met (if any)
    :rtype:   - filename_out: str
              - conditions_not_met: list
    """

    # enable standalone code generation (see brian2 doc: computational methods & efficiency)
    if STANDALONE_CODE_GEN:
        b2.set_device('cpp_standalone')

    # check inputs
    assert isinstance(model_name, str), "argument 'model_name' must be a string"
    assert config_preloaded is None or isinstance(config_preloaded, dict), \
        "argument 'config_preloaded' must be either None or dictionary (same format as if loaded from config .json)"
    assert sim_time_ms is None or (isinstance(sim_time_ms, int) and sim_time_ms >= 0), \
        "argument 'sim_time_ms' must be an integer >= 0 or None"
    assert isinstance(clock_dt_ms, float) and 0 < clock_dt_ms <= 1, \
        "argument 'clock_dt_ms' must be a float > 0 and <= 1"
    assert log_name is None or isinstance(log_name, str), \
        "argument 'log_name' must be either None or string (name of logging.Logger instance)"
    assert filename_out_arg is None or isinstance(filename_out_arg, str), \
        "argument 'filename_out_arg' must be either None or string (filename of output .pkl file)"
    assert isinstance(run_id, int), "argument run_id must be an integer"

    # if no logger name was passed, create one for console output, otherwise get the one identified by the passed name
    if not log_name:
        log_name = 'default_console'
        log = logging.getLogger(log_name)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(console_handler)
    else:
        log = logging.getLogger(log_name)

    # load model config file (.json), or use passed config
    if config_preloaded:
        assert filename_out_arg, 'if preloaded config is passed, filename_out_arg must be passed as well'
        config = config_preloaded
        filename_cfg = None
        filename_out = filename_out_arg
        log.info("Preloaded config used. filename_out: " + filename_out)
    else:
        config, filename_cfg, filename_out = p_io.load_config(model_name)
        log.info("Config file '" + filename_cfg + "' loaded.")
        if filename_out_arg:
            log.info("o default filename_out is overwritten from " + filename_out + " to " + filename_out_arg)
            filename_out = filename_out_arg

    # get seed for random number generator. load from config if present. if single value, duplicate for noise and conn
    if 'rng_seed' in config['misc']:
        assert isinstance(config['misc']['rng_seed'], list) or isinstance(config['misc']['rng_seed'], int), \
            "config['misc']['rng_seed'] must be an int or a list"
        if isinstance(config['misc']['rng_seed'], list):
            assert len(config['misc']['rng_seed']) == 5, "config['misc']['rng_seed'] must be an int or a list of len 5"
            rng_seed = config['misc']['rng_seed']
        else:
            rng_seed = [config['misc']['rng_seed']] * 5
        log.info("Using rng seeds from config: " + str(rng_seed))
    else:
        rng_seed = [1, 1, 1, 1, 1]
        log.info("Using default rng seeds: " + str(rng_seed))

    # set brian2 default clock time step for simulation
    b2.defaultclock.dt = clock_dt_ms * b2.ms

    # build differential equation of Leaky Integrate-and-Fire model from config
    eqtn = b2.Equations('')
    for eq in range(len(config['ode'])):
        eqtn += b2.Equations(config['ode'][eq])
    log.info("Equation loaded")
    if b_verbose:
        log.info(eqtn)

    # misc parameters
    refractory_period = config['misc']['refractory_period'] * b2.ms
    population_id = config['misc']['population_id']
    n_populations = len(population_id)
    n_per_pop = config['misc']['n_neurons_per_pop']
    n_neurons = sum(n_per_pop)
    n_synapses = len(config['synapse']['syn_pre_idx'])
    if sim_time_ms is None:
        sim_time_ms = config['misc']['sim_time']
    if 'input_current' in config and 'power_ramp' in config['input_current']:
        b_power_ramp = config['input_current']['power_ramp']
    else:
        b_power_ramp = '[not in cfg]'
    log.info("--- misc parameters ---\n- Refractory period: " + str(refractory_period) +
             "\n- N populations: " + str(n_populations) + "\n- N neurons per population: " + str(n_per_pop) +
             "\n- Simulation duration: " + str(sim_time_ms) + "\n- defaultclock.dt: " + str(b2.defaultclock.dt) +
             "\n- Power ramp: " + str(b_power_ramp) + "\n-----------------------")

    # # # # # # # # #
    # INPUT CURRENT #
    # # # # # # # # #

    # seed rng for input noise offset generation
    b2.seed(int(rng_seed[0]))

    # get a set of random numbers for continuous amplitude offsets in current generation
    offset_noise = p_input.get_offset_noise(config)
    if any(np.concatenate(offset_noise)):
        n_offset_values_per_pop = [len(offset_noise[i]) for i in range(len(offset_noise))]
        if sum(n_offset_values_per_pop) > 0:
            log.info("~ " + str(n_offset_values_per_pop) + " random numbers generated for" +
                     " noise offset in input current generation")

    # re-seed rng for white noise generation
    b2.seed(int(rng_seed[1]))

    # get a set of random numbers for white noise to be added to current amplitudes
    white_noise = p_input.get_white_noise(config)
    if any(white_noise):
        n_wnoise_values_per_pop = [len(white_noise[i]) for i in range(len(white_noise))]
        if sum(n_wnoise_values_per_pop) > 0:
            log.info("~ " + str(n_wnoise_values_per_pop) + " random numbers generated for" +
                     " white noise in input current generation")

    # re-seed rng for random initial voltage generation
    b2.seed(int(rng_seed[2]))

    # get a set of random numbers for initial voltage between resting potential and spiking threshold
    initial_v_noise = [None] * n_populations
    if 'rand_initial_v' in config['misc']:
        for pop in range(n_populations):
            if config['misc']['rand_initial_v'][pop]:
                thresh_tmp = config['parameters_nrn']['v_thresh'][pop]
                rest_tmp = config['parameters_nrn']['El'][pop]
                initial_v_noise[pop] = b2.rand(n_per_pop[pop]) * (thresh_tmp - rest_tmp) + rest_tmp

    # if there's a non-zero parameter "fI_amp_stepsize" in "input_current", generate a range of current amplitudes, each
    # feeding into a different neuron population (first neuron population will be cloned here as many times as needed)
    # NOTE: This overwrites values in config.
    if 'fI_amp_stepsize' in config['input_current'] and config['input_current']['fI_amp_stepsize']:
        assert len(config['input_current']['amp_start'][0]) == 2, "when providing fI_amp_stepsize for " + \
                                                                  "input_current, amp_start must have two entries " + \
                                                                  "(lower and upper bound)"
        lower_bound = config['input_current']['amp_start'][0][0]
        upper_bound = config['input_current']['amp_start'][0][1]
        difference = upper_bound - lower_bound
        amp_steps = np.linspace(lower_bound, upper_bound,
                                round(difference / config['input_current']['fI_amp_stepsize']) + 1)
        n_steps = len(amp_steps)
        config['input_current']['t_start'] = [config['input_current']['t_start'][0]] * n_steps
        config['input_current']['duration'] = [config['input_current']['duration'][0]] * n_steps
        config['input_current']['amp_start'] = [[amp_val] for amp_val in amp_steps]
        config['input_current']['amp_end'] = [[] for _ in range(n_steps)]
        config['input_current']['offset_range'] = [config['input_current']['offset_range'][0]] * n_steps
        log.info("~ generated " + str(n_steps) + " amplitude steps for input current: " +
                 ' '.join([str(round(amp_val, 3)) for amp_val in amp_steps]))
        # if number of populations is smaller than number of steps (could be 1), correct the number of populations
        if n_populations < n_steps:
            config['misc']['population_id'] = [population_id[0]] * n_steps
            offset_noise = offset_noise * n_steps
            white_noise = white_noise * n_steps
            population_id = config['misc']['population_id']
            n_populations = len(population_id)
            config['misc']['n_neurons_per_pop'] = [n_per_pop[0]] * n_steps
            n_per_pop = config['misc']['n_neurons_per_pop']
            n_neurons = sum(n_per_pop)
            log.info("~ number of populations extended to " + str(n_populations) + " with " + str(n_per_pop[0]) +
                     " neuron(s) per population each, for a total of " + str(n_neurons) + " neurons")

    # # # # # #
    # NEURONS #
    # # # # # #

    # create neurons (method must be changed for stochastic models)
    t_neurongroup_start = timer()
    if 'integration_method' in config['misc'] and config['misc']['integration_method']:
        neurons = b2.NeuronGroup(n_neurons, model=eqtn, threshold='v > v_thresh', reset='v = v_reset',
                                 refractory=refractory_period, method=config['misc']['integration_method'])
    else:
        neurons = b2.NeuronGroup(n_neurons, model=eqtn, threshold='v > v_thresh', reset='v = v_reset',
                                 refractory=refractory_period)
    t_neurongroup_end = timer()
    log.info("NeuronGroup (n=" + str(n_neurons) + ") created in "
             + str(round(t_neurongroup_end - t_neurongroup_start, 1)) + " seconds.")

    # create subgroups (populations)
    populations = []
    for pop in range(n_populations):
        n_so_far = sum(n_per_pop[0:pop])
        populations.append(neurons[n_so_far:n_so_far + n_per_pop[pop]])

    # # # # # # #
    # SYNAPSES  #
    # # # # # # #

    # re-seed rng for neuron connectivity
    b2.seed(int(rng_seed[3]))

    # build equation of synapse model from config
    syn_eqtn_model = b2.Equations('')
    for eq in range(len(config['synapse']['model'])):
        syn_eqtn_model += b2.Equations(config['synapse']['model'][eq])
    if b_verbose:
        log.info("Synapse equation loaded: " + str(syn_eqtn_model))

    # load synapse on_pre equation(s), connection indices, probabilities and conditions from config
    syn_eqtn_on_pre = config['synapse']['on_pre']
    syn_pre_idx = config['synapse']['syn_pre_idx']
    syn_post_idx = config['synapse']['syn_post_idx']
    if 'connection_probability' in config['synapse']:
        connection_probability = config['synapse']['connection_probability']
    else:
        connection_probability = [1] * n_synapses
    if 'condition' in config['synapse']:
        syn_condition = config['synapse']['condition']
    else:
        syn_condition = [""] * n_synapses

    # check for multiple on_pre equations and load indices per synapse from config
    if type(syn_eqtn_on_pre) is list:
        if 'on_pre_idx_syn' in config['synapse'] and len(config['synapse']['on_pre_idx_syn']) == n_synapses:
            on_pre_idx_syn = config['synapse']['on_pre_idx_syn']
            log.info("- on_pre equations:" + str(syn_eqtn_on_pre) + " with indices per synapse: " +
                     str(on_pre_idx_syn))
        else:
            log.error("ERROR: Multiple equations in 'on_pre' require list 'on_pre_idx_syn' in 'synapse'" +
                      " with number of elements equal to number of synapses (i.e. len(syn_pre_idx))")
            raise ValueError("Multiple equations in 'on_pre' require list 'on_pre_idx_syn' in 'synapse'" +
                             " with number of elements equal to number of synapses (i.e. len(syn_pre_idx))")
    else:
        syn_eqtn_on_pre = [syn_eqtn_on_pre]
        on_pre_idx_syn = [0] * n_synapses
        log.info("- on_pre equation:" + str(syn_eqtn_on_pre) + " for all " + str(n_synapses) + " synapses")

    # check length of parameters
    if n_synapses > 0:
        if not (len(syn_pre_idx) is len(syn_post_idx) is len(on_pre_idx_syn) is len(syn_condition) is
                len(connection_probability)):
            log.error("syn_pre_idx, syn_post_idx, on_pre_idx_syn, syn_condition & connection_probability must have " +
                      "the same length (can contain empty elements)")
            raise ValueError("syn_pre_idx, syn_post_idx, on_pre_idx_syn, syn_condition & connection_probability must" +
                             "have the same length (can contain empty elements)")
    # create synapses
    synapses = []
    t_synapses_start = timer()
    for syn in range(n_synapses):
        synapses.append(b2.Synapses(populations[syn_pre_idx[syn]], populations[syn_post_idx[syn]],
                                    model=syn_eqtn_model, on_pre=syn_eqtn_on_pre[on_pre_idx_syn[syn]]))
        if syn_condition[syn]:
            synapses[syn].connect(condition=syn_condition[syn], p=connection_probability[syn])
        else:
            synapses[syn].connect(p=connection_probability[syn])
    t_synapses_end = timer()
    log.info("Synapses (n=" + str(n_synapses) + ") created in " + str(round(t_synapses_end - t_synapses_start, 1))
             + " seconds.")

    # copy synaptic connection indices etc. for later saving
    connectivity = []
    for syn in range(n_synapses):
        connectivity.append(synapses[syn].get_states(['i', 'j', 'N_incoming', 'N_outgoing']))

    # # # # # # # #
    # GENERATORS  #
    # # # # # # # #

    # re-seed rng for generator connectivity and poisson groups
    b2.seed(int(rng_seed[4]))

    # create spike generators from passed input spiketimes to feed into model neurons
    generators = []
    n_nrns_per_gen = []
    connectivity_generator = []
    if input_spiketimes_ms:
        t_spikegen_start = timer()
        n_generators = len(input_spiketimes_ms)
        for gen in range(n_generators):
            n_nrns_per_gen.append(len(input_spiketimes_ms[gen]))
            n_spikes_per_gen_nrn = [len(input_spiketimes_ms[gen][nrn]) for nrn in range(len(input_spiketimes_ms[gen]))]
            generator_indices = np.arange(n_nrns_per_gen[gen])
            generators.append(b2.SpikeGeneratorGroup(n_nrns_per_gen[gen],
                                                     np.repeat(generator_indices, n_spikes_per_gen_nrn),
                                                     np.concatenate(input_spiketimes_ms[gen], axis=None) * b2.ms))

        # check for generator synapse indices in config
        if 'generator' in config and 'gen_pre_idx' in config['generator'] and config['generator']['gen_pre_idx']:
            gen_pre_idx = config['generator']['gen_pre_idx']
            gen_post_idx = config['generator']['gen_post_idx']
            log.info("| generator synaptic indices loaded from config: " + str(gen_pre_idx) + "->" + str(gen_post_idx))
        else:
            # if none provided in config, connect one-to-one or all-to-all depending on number of generators/populations
            if n_generators == n_populations:
                gen_pre_idx = list(range(n_generators))
                gen_post_idx = list(range(n_populations))
                log.info("| generator synaptic indices one-to-one: " + str(gen_pre_idx) + "->" + str(gen_post_idx))
            elif n_populations == 1:
                gen_pre_idx = list(range(n_generators))
                gen_post_idx = [0] * n_generators
                log.info("| generator synaptic indices all-to-one: " + str(gen_pre_idx) + "->" + str(gen_post_idx))
            else:
                gen_pre_idx = list(range(n_generators)) * n_populations
                tmp_post_idx = []
                [tmp_post_idx.append([pop] * n_generators) for pop in range(n_populations)]
                gen_post_idx = np.concatenate(tmp_post_idx).tolist()
                log.info("| generator synaptic indices all-to-all: " + str(gen_pre_idx) + "->" + str(gen_post_idx))
        n_gen_synapses = len(gen_pre_idx)

        # load generator on_pre equation(s), connection probabilities and conditions etc. from config
        gen_eqtn_on_pre = config['synapse']['on_pre']  # note: gen_eqtn_on_pre is equal to syn_eqtn_on_pre
        if 'connection_probability' in config['generator']:
            gen_connection_probability = config['generator']['connection_probability']
        else:
            gen_connection_probability = [1] * n_gen_synapses
        if 'condition' in config['generator']:
            gen_condition = config['generator']['condition']
        else:
            gen_condition = [""] * n_gen_synapses

        # check for multiple on_pre equations and load indices per generator from config
        if type(gen_eqtn_on_pre) is list:
            if 'generator' in config and 'on_pre_idx_gen' in config['generator'] \
                    and len(config['generator']['on_pre_idx_gen']) == n_gen_synapses:
                on_pre_idx_gen = config['generator']['on_pre_idx_gen']
                log.info("- on_pre equations:" + str(gen_eqtn_on_pre) + " with indices per generator: " +
                         str(on_pre_idx_gen))
            else:
                log.error("ERROR: Multiple equations in on_pre require list on_pre_idx_gen in generator, with number" +
                          " of elements equal to number of generators (i.e. len(input_spiketimes_ms))")
                raise ValueError("Multiple equations in on_pre require list on_pre_idx_gen in generator, with number" +
                                 " of elements equal to number of generators (i.e. len(input_spiketimes_ms))")
        else:
            gen_eqtn_on_pre = [gen_eqtn_on_pre]
            on_pre_idx_gen = [0] * n_gen_synapses
            log.info("- on_pre equation:" + str(gen_eqtn_on_pre) + " for all " + str(n_generators) + " generators")

        # create synapses from generators onto model neurons
        synapses_generator = []
        for gen in range(n_gen_synapses):
            synapses_generator.append(b2.Synapses(generators[gen_pre_idx[gen]], populations[gen_post_idx[gen]],
                                                  model=syn_eqtn_model, on_pre=gen_eqtn_on_pre[on_pre_idx_gen[gen]]))
            if 'generator' in config and gen_condition[gen]:
                synapses_generator[gen].connect(condition=gen_condition[gen], p=gen_connection_probability[gen])
                log.info(" connection condition for generators loaded from config: " + gen_condition[gen])
            else:
                # default if no condition was supplied in config file
                synapses_generator[gen].connect(p=gen_connection_probability[gen])
        t_spikegen_end = timer()
        log.info("Spikegenerators (n=" + str(n_generators) + ") and their synapses (n=" + str(n_gen_synapses) +
                 ") created in " + str(round(t_spikegen_end - t_spikegen_start, 1)) + " seconds.")

        # copy spikegenerator connection indices etc. for later saving
        for gen in range(n_generators):
            connectivity_generator.append(synapses_generator[gen].get_states(['i', 'j', 'N_incoming', 'N_outgoing']))

    # # # # # # # # #
    # POISSON GROUP #
    # # # # # # # # #

    # generate poisson group, simulate once and record the spikes into a SpikeGenerator, to be used in all runs
    if 'poisson_group' in config and config['poisson_group']:
        # get timed array containing the spike rate for the poisson group at each timestep in the simulation
        timer_poisson_start = timer()
        poisson_rate = p_input.get_rates_for_poisson_group(config, log_name, sim_time_ms=sim_time_ms)

        # generate poisson group and spike monitor
        n_poisson_neurons = config['poisson_group']['n_neurons']
        poisson = b2.PoissonGroup(N=n_poisson_neurons, rates='poisson_rate(t)')  # string = name of timedarray
        spikemon_poisson = b2.SpikeMonitor(poisson)

        # run simulation with poisson group once and record spikes
        net_poisson = b2.Network(poisson, spikemon_poisson)
        net_poisson.run(sim_time_ms * b2.ms)

        # construct spike generator with spikes created by the poisson group
        generator_poisson = b2.SpikeGeneratorGroup(n_poisson_neurons, spikemon_poisson.i, spikemon_poisson.t)

        # save recorded spikes and delete group and monitor (should not get added to subsequent networks)
        spikes_poisson = spikemon_poisson.get_states()
        del poisson
        del spikemon_poisson

        # connect the generator to the neuron populations given by config['poisson_group']['syn_post_idx']
        psn_syn_post_idx = config['poisson_group']['syn_post_idx']
        psn_on_pre_idx_syn = config['poisson_group']['on_pre_idx_syn']
        n_psn_synapses = len(psn_syn_post_idx)
        synapses_poisson = []
        for psn in range(n_psn_synapses):
            synapses_poisson.append(b2.Synapses(generator_poisson, populations[psn_syn_post_idx[psn]],
                                                model=syn_eqtn_model, on_pre=syn_eqtn_on_pre[psn_on_pre_idx_syn[psn]]))
            # note: syn_condition for poisson group not implemented
            synapses_poisson[psn].connect(p=config['poisson_group']['connection_probability'][psn])
        timer_poisson_end = timer()
        log.info("Poisson Group (n=" + str(n_poisson_neurons) + ") spikes and synapses (n=" + str(n_psn_synapses)
                 + ") created in " + str(round(timer_poisson_end - timer_poisson_start, 1)) + " seconds.")
    else:
        spikes_poisson = None
        generator_poisson = None

    # # # # # # # # # # # # #
    # MONITORS AND NETWORK  #
    # # # # # # # # # # # # #

    # set up state monitor (continuously record variables) and spike monitor (record spike times)
    spike_cur_run = b2.SpikeMonitor(neurons)
    state_cur_run = b2.StateMonitor(neurons, recorded_variables, record=True)
    log.info("o recorded variables: " + str(recorded_variables))

    # create network and add synapses, as they are  not automatically collected, because hidden in a container (list)
    net = b2.Network(b2.collect())
    net.add(synapses)
    if input_spiketimes_ms:
        net.add(synapses_generator)
        net.add(generators)
    if generator_poisson:
        net.add(generator_poisson)
        net.add(synapses_poisson)

    # store network status (will be restored at the beginning of each run)
    net.store()

    # # # # # # # # # #
    # FREE PARAMETER  #
    # # # # # # # # # #

    # get name of free parameter, stepsize and dict containing it e.g. "parameters_nrn"; if the config file contains one
    # note: restricted to a single parameter atm!
    free_parameter_keys = list(config['free_parameter_stepsize'].keys())
    if not free_parameter_keys:
        free_parameter_name = None
        free_parameter_stepsize = None
        free_parameter_dict = None
    else:
        free_parameter_name = free_parameter_keys[0]
        free_parameter_stepsize = config['free_parameter_stepsize'][free_parameter_name]
        if free_parameter_name in config['parameters_nrn'] and config['parameters_nrn'][free_parameter_name] \
                and isinstance(config['parameters_nrn'][free_parameter_name][0], list):
            free_parameter_dict = 'parameters_nrn'
        elif free_parameter_name in config['parameters_syn'] and config['parameters_syn'][free_parameter_name] \
                and isinstance(config['parameters_syn'][free_parameter_name][0], list):
            free_parameter_dict = 'parameters_syn'
        elif 'parameters_gen' in config and free_parameter_name in config['parameters_gen'] \
                and config['parameters_gen'][free_parameter_name] \
                and isinstance(config['parameters_gen'][free_parameter_name][0], list):
            free_parameter_dict = 'parameters_gen'
        elif free_parameter_name in config['input_current']:
            free_parameter_dict = 'input_current'
        else:
            log.error("ERROR: Free parameter name '" + free_parameter_name + "' not found in dictionaries " +
                      "'parameters_nrn', 'parameters_syn', 'parameters_gen' or 'input_current' in " + model_name)
            raise ValueError("Free parameter name '" + free_parameter_name + "' not found in dictionaries " +
                             "'parameters_nrn', 'parameters_syn', 'parameters_gen' or 'input_current' in " + model_name)

    if free_parameter_name:
        # get all possible values for the free parameter (fp_values) within the range (lower_bound to upper_bound)
        # for the groups (populations or synapses) given in the config file.
        fp_values = []
        # track for which group the free parameter changes (i.e. which input currents must be re-generated across runs)
        b_fp_per_group = [0] * len(config[free_parameter_dict][free_parameter_name][0])
        for group in range(len(config[free_parameter_dict][free_parameter_name][0])):
            if type(config[free_parameter_dict][free_parameter_name][0][group]) is not list:
                lower_bound = config[free_parameter_dict][free_parameter_name][0][group]
                upper_bound = config[free_parameter_dict][free_parameter_name][1][group]
                difference = upper_bound - lower_bound
                b_fp_per_group[group] = difference > 0
                fp_values.append(np.linspace(lower_bound, upper_bound, round(difference / free_parameter_stepsize) + 1))
            else:
                # if the free parameter contains three levels of nested lists (i.e. multiple values per group, like
                # some of the input_current parameters), loop through an additional level.
                fp_values.append([])
                for subval in range(len(config[free_parameter_dict][free_parameter_name][0][group])):
                    assert type(config[free_parameter_dict][free_parameter_name][0][group][subval]) is not list, \
                        "too many levels in nested lists of free parameter " + free_parameter_name + "? (max: [[[]]])"
                    lower_bound = config[free_parameter_dict][free_parameter_name][0][group][subval]
                    upper_bound = config[free_parameter_dict][free_parameter_name][1][group][subval]
                    difference = upper_bound - lower_bound
                    b_fp_per_group[group] = difference > 0
                    fp_values[group].append(np.linspace(lower_bound, upper_bound,
                                                        round(difference / free_parameter_stepsize) + 1))

        # if the free parameter is varied for multiple populations, 'free_parameter_combination_type' determines
        # whether all possible permutations should be simulated ('product', e.g. [1,1], [1,2], [2,1], [2,2])
        # or whether values should be iterated in series ('serial', e.g. [1,1], [2,2])
        if 'free_parameter_combination_type' not in config or not config['free_parameter_combination_type'] \
                or config['free_parameter_combination_type'][0] == "product":
            assert type(fp_values[0]) is not list, "'product' combination not implemented for free parameters " + \
                                                   "containing nested lists (e.g. input_current: t_start). Set " + \
                                                   "'free_parameter_combination_type' to 'serial' in config .json."
            # get a list of all combinations of these values
            combinations = list(itertools.product(*fp_values))
        elif config['free_parameter_combination_type'][0] == "serial":
            if type(fp_values[0]) is not list:
                maxlen = max([fp_values[i].size for i in range(len(fp_values))])
                values_per_group = []  # this will contain repetitions of the values that are not iterated
                for group in range(len(fp_values)):
                    if fp_values[group].size < maxlen:
                        values_per_group.append(tuple(fp_values[group].tolist() +
                                                      (maxlen - fp_values[group].size) * [fp_values[group][-1]]))
                    else:
                        values_per_group.append(tuple(fp_values[group]))
                # get a list of len(n_runs) with each sublist containing the parameter values for that run
                combinations = [list(tup) for tup in zip(*values_per_group)]
            else:
                # if the free param contains three levels of nested lists, loop through an additional level (see above)
                length_of_sublists = [[len(s) for s in g] for g in fp_values]
                maxlen = max(itertools.chain.from_iterable(length_of_sublists))  # maximum in flattened list
                values_per_group = []  # this will contain repetitions of the values that are not iterated
                for group in range(len(fp_values)):
                    values_per_group.append([])
                    for subval in range(len(fp_values[group])):
                        if fp_values[group][subval].size < maxlen:
                            values_per_group[group].append(tuple(fp_values[group][subval].tolist() +
                                                                 (maxlen - fp_values[group][subval].size) *
                                                                 [fp_values[group][subval][-1]]))
                        else:
                            values_per_group[group].append(tuple(fp_values[group][subval]))
                # get a list of len(n_runs) with each sublist containing the parameter values for that run
                combinations = []
                for i_combination in range(maxlen):
                    combinations.append([])
                    for group in range(len(values_per_group)):
                        combinations[i_combination].append([])
                        for subval in range(len(values_per_group[group])):
                            combinations[i_combination][group].append(values_per_group[group][subval][i_combination])
        else:
            log.error("ERROR: 'free_parameter_combination_type' in .json should be either 'product', 'serial' or empty")
            raise ValueError("'free_parameter_combination_type' in .json should be either 'product', 'serial' or empty")
        # number of simulation runs (one per combination of the free parameter)
        n_runs = len(combinations)
    else:
        n_runs = 1

    # # # # # # # # # #
    # RUN SIMULATION  #
    # # # # # # # # # #

    # make a copy of the config dictionary to update the free parameter for each run in the following loop.
    config_cur_run = copy.deepcopy(config)

    # run simulation for each combination of the free parameter (if any)
    states = []
    spikes = []
    info = []
    for r in range(n_runs):
        # get combination of values that is used for the free parameter in this run and get them into the copy
        # of the model config dictionary
        if free_parameter_name:
            config_cur_run[free_parameter_dict][free_parameter_name] = combinations[r]

        # restore network status
        net.restore()

        # re-set spikes in spikegenerator. workaround for bug that got fixed in Brian2 v2.3, see here:
        # https://github.com/brian-team/brian2/issues/1084  todo: can be removed if using Brian2 version >= 2.3
        if input_spiketimes_ms:
            for gen in range(n_generators):
                generators[gen].set_spikes(np.repeat(generator_indices, n_spikes_per_gen_nrn),
                                           np.concatenate(input_spiketimes_ms[gen], axis=None) * b2.ms)

        # generate input currents
        if r == 0:
            I_input, log_msg_curr = p_input.setup_currents(config_cur_run, offset_noise, white_noise, log_name,
                                                           sim_time_ms=sim_time_ms)
            log.info(log_msg_curr)
        elif free_parameter_dict == 'input_current':
            I_input, log_msg_curr = p_input.setup_currents(config_cur_run, offset_noise, white_noise, log_name,
                                                           I_input_prev=I_input, sim_time_ms=sim_time_ms,
                                                           b_gen_curr_per_pop=b_fp_per_group)
            log.info(log_msg_curr)

        # get parameters and initial states and their corresponding units from config_cur_run
        # El: Resting potential
        # Rm: Membrane resistance
        # Ie: Input current
        # Is: Synaptic current
        # taum: membrane time constant
        # v_thresh: threshold potential for spike generation
        # v_reset: reset potential after spike
        # tau1, tau2: synaptic time constants
        for param, val in config_cur_run['parameters_nrn'].items():
            if param in config_cur_run['parameter_units']:
                if hasattr(b2.units, config_cur_run['parameter_units'][param]):
                    unit = 'b2.' + config_cur_run['parameter_units'][param]
                else:
                    log.error("ERROR: Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                              "' for nrn parameter '" + param + "' in " + model_name)
                    raise ValueError("Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                     "' for nrn parameter '" + param + "' in " + model_name)
            else:
                unit = '1'
                log.info("* no unit found in " + model_name + " for nrn parameter " + param + " => assigning 1")
            # if one value is given but multiple populations, repeat that value for all populations
            if len(val) == 1 and n_populations > 1:
                log.info("* single value " + str(val) + " given for parameter " + param +
                         " => using for all populations")
                val = val * n_populations
                config_cur_run['parameters_nrn'][param] = val
            for pop in range(n_populations):
                setattr(populations[pop], param, val[pop] * eval(unit))
                if b_verbose:
                    log.info("- parameter '" + param + "' for population " + str(pop) + " set to <" + str(val[pop]) +
                             "> with unit " + unit)
            # randomize initial conditions
            if param == 'v' and 'rand_initial_v' in config['misc']:
                for pop in range(n_populations):
                    if initial_v_noise[pop] is not None:
                        for nrn in range(len(populations[pop].v)):
                            populations[pop].v[nrn] = initial_v_noise[pop][nrn] * eval(unit)

        # get synaptic parameters and their respective units from config_cur_run
        # w: Synaptic weight
        # delay: Synaptic delay
        for param, val in config_cur_run['parameters_syn'].items():
            if param in config_cur_run['parameter_units']:
                if hasattr(b2.units, config_cur_run['parameter_units'][param]):
                    unit = 'b2.' + config_cur_run['parameter_units'][param]
                else:
                    log.error("ERROR: Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                              "' for syn parameter '" + param + "' in " + model_name)
                    raise ValueError("Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                     "' for syn parameter '" + param + "' in " + model_name)
            else:
                unit = '1'
                log.info("* no unit found in " + model_name + " for syn parameter " + param + " => assigning 1")
            # if one value is given but multiple populations, repeat that value for all populations
            if len(val) == 1 and n_synapses > 1:
                log.info("* single value " + str(val) + " given for synaptic parameter " + param +
                         " => using for all synapses")
                val = val * n_synapses
                config_cur_run['parameters_syn'][param] = val
            for syn in range(n_synapses):
                setattr(synapses[syn], param, val[syn] * eval(unit))

        # get synaptic parameters for spike generator
        if input_spiketimes_ms:
            for param, val in config_cur_run['parameters_gen'].items():
                if param in config_cur_run['parameter_units']:
                    if hasattr(b2.units, config_cur_run['parameter_units'][param]):
                        unit = 'b2.' + config_cur_run['parameter_units'][param]
                    else:
                        log.error("ERROR: Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                  "' for syn parameter '" + param + "' in " + model_name)
                        raise ValueError("Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                         "' for gen parameter '" + param + "' in " + model_name)
                else:
                    unit = '1'
                    log.info("* no unit found in " + model_name + " for generator parameter " + param +
                             " => assigning 1")
                # if one value is given but multiple populations, repeat that value for all populations
                if len(val) == 1 and n_gen_synapses > 1:
                    log.info("* single value " + str(val) + " given for generator parameter " + param +
                             " => using for all generator connections")
                    val = val * n_gen_synapses
                    config_cur_run['parameters_gen'][param] = val
                for gen in range(n_gen_synapses):
                    setattr(synapses_generator[gen], param, val[gen] * eval(unit))

        # get synaptic parameters for poisson group
        if generator_poisson:
            for param, val in config_cur_run['parameters_psn'].items():
                if param in config_cur_run['parameter_units']:
                    if hasattr(b2.units, config_cur_run['parameter_units'][param]):
                        unit = 'b2.' + config_cur_run['parameter_units'][param]
                    else:
                        log.error("ERROR: Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                  "' for psn parameter '" + param + "' in " + model_name)
                        raise ValueError("Unrecognized unit '" + config_cur_run['parameter_units'][param] +
                                         "' for psn parameter '" + param + "' in " + model_name)
                else:
                    unit = '1'
                    log.info("* no unit found in " + model_name + " for poisson group parameter " + param +
                             " => assigning 1")
                # if one value is given but multiple populations, repeat that value for all populations
                if len(val) == 1 and n_psn_synapses > 1:
                    log.info("* single value " + str(val) + " given for poisson group parameter " + param +
                             " => using for all poisson group connections")
                    val = val * n_psn_synapses
                    config_cur_run['parameters_psn'][param] = val
                for psn in range(n_psn_synapses):
                    setattr(synapses_poisson[psn], param, val[psn] * eval(unit))

        # run and time simulation
        t_sim_start = timer()
        if recording_start_ms > 0 and recording_start_ms < sim_time_ms:
            log.info("o starting recording spikes and states at " + str(recording_start_ms) + "ms into the simulation")
            state_cur_run.active = False
            spike_cur_run.active = False
            net.run(recording_start_ms * b2.ms)
            state_cur_run.active = True
            spike_cur_run.active = True
            net.run((sim_time_ms - recording_start_ms) * b2.ms)
        else:
            net.run(sim_time_ms * b2.ms)
        # net.run(sim_time_ms * b2.ms, profile=True)
        # print(b2.profiling_summary(net, show=10))
        run_time = timer() - t_sim_start

        # output run time (and values for free parameter, if any)
        if free_parameter_name:
            log.info("Run " + str(r) + " finished in " + str(round(run_time, 1)) + " seconds. Values for free param '" +
                     free_parameter_name + "': " + str(combinations[r]))
        else:
            log.info("Run " + str(r) + " finished in " + str(round(run_time, 1)) + " seconds.")

        # check which parameters are available for info dict
        syn_weights = []
        if 'w' in config_cur_run['parameters_syn']:
            syn_weights = config_cur_run['parameters_syn']['w']

        # create dictionary containing additional information about simulation
        if free_parameter_name:
            free_parameter_values = combinations[r]
        else:
            free_parameter_values = ()
        info.append({'filename': model_name,
                     'run_id': run_id,
                     'run_date': datetime.now(),
                     'run_time': run_time,
                     'n_populations': len(populations),
                     'population_ids': population_id,
                     'population_sizes': [pop.N for pop in populations],
                     'generator_sizes': [gen.N for gen in generators],
                     'sim_time': sim_time_ms,
                     'dt': b2.defaultclock.dt,
                     'v_thresh': neurons.v_thresh / b2.mV,
                     'syn_pre_idx': syn_pre_idx,
                     'syn_post_idx': syn_post_idx,
                     'syn_weight': syn_weights,
                     'syn_delay': config_cur_run['parameters_syn']['delay'],
                     'free_parameter_name': free_parameter_name,
                     'free_parameter_dict': free_parameter_dict,
                     'free_parameter_values': free_parameter_values,
                     'connection_probability': connection_probability})

        # save states and spikes into lists
        states.append(state_cur_run.get_states())
        spikes.append(spike_cur_run.get_states())

        # end for r in range(n_runs)

    # build standalone code
    if STANDALONE_CODE_GEN:
        b2.device.build(directory='output', compile=True, run=True, debug=False)

    # save results
    if not b_always_save_output:
        conditions_not_met = is_not_worth_saving(spikes[0], states[0], info[0], config)
    else:
        conditions_not_met = []
    if b_always_save_output or not any(conditions_not_met):
        msg = p_io.save_monitors(states, spikes, spikes_poisson, connectivity, connectivity_generator, config, info,
                                 filename_out)
        log.info(msg)

    # copy config file contents to log file (if there is one)
    file_hndlr = [log.handlers[i] for i in range(len(log.handlers)) if isinstance(log.handlers[i], logging.FileHandler)]
    if file_hndlr:
        if filename_cfg:
            with open(file_hndlr[0].baseFilename, 'a') as file_log, open(filename_cfg, 'r') as file_cfg:
                file_log.write("\n======= " + filename_cfg + " =======\n")
                for line in file_cfg:
                    file_log.write(line)
        else:
            with open(file_hndlr[0].baseFilename, 'a') as file_log:
                file_log.write("\n======= preloaded config =======\n")
                json.dump(config, file_log, indent=4)
                file_log.write("\n======= end =======\n\n")

    return filename_out, conditions_not_met


def is_not_worth_saving(spikemon, statemon, info, config):
    """Check certain conditions for whether to save the ouput of the current simulation, based on some of its results.

    :param spikemon: dict containing SpikeMonitor data from b2...get_states()
    :type spikemon: dict
    :param statemon: dict containing StateMonitor data from b2...get_states()
    :type statemon: dict
    :param info: dictionary containing additional information about simulation (as created by run_simulation())
    :type info: dict
    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :return:  conditions_not_met: list of indices of conditions not met (if any)
    :rtype:   conditions_not_met: list
    """

    # get neuron of interest from config
    nrn_oi_abs = p_util.get_abs_from_rel_nrn_idx(config['plot']['idx_nrn_oi_relative'], config['plot']['idx_pop_oi'],
                                                 config['misc']['n_neurons_per_pop'])
    # assure that neuron of interest spikes at all, doesn't spike until a certain time, has a maximum number of spikes
    # and doesn't cross a minimum membrane potential
    min_t_first_spike_ms = 100 * b2.ms
    max_n_spikes = 6
    min_voltage = -100
    b_spike_from_nrn_oi = spikemon['i'] == nrn_oi_abs
    n_spikes_nrn_oi = sum(b_spike_from_nrn_oi)
    spiketimes_int = np.concatenate(np.array(p_util.get_spiketimes_from_monitor(spikemon, info, 1)))
    t1 = 50
    t2 = 100
    i1 = np.where(statemon['t'] / b2.ms >= t1)[0][0]
    i2 = np.where(statemon['t'] / b2.ms >= t2)[0][0]
    n_int_spikes = sum(np.logical_and(spiketimes_int > t1, spiketimes_int < t2)) / config['misc']['n_neurons_per_pop'][1]
    spikerate_int = n_int_spikes / (t2 - t1) * 1000
    baseline_pm = np.mean(statemon['v'][i1:i2, nrn_oi_abs] / b2.mV)
    baseline_std_pm = np.std(statemon['v'][i1:i2, nrn_oi_abs] / b2.mV)
    # check for conditions not met and add their ids to the returned list
    conditions_not_met = []
    # if not any(b_spike_from_nrn_oi):
    #    conditions_not_met += [1]  # nrn_oi_spikes
    # if any(statemon['v'][:, nrn_oi_abs] / b2.mV < min_voltage):
    #     conditions_not_met += [2]  # nrn_oi_min_v
    # if any(b_spike_from_nrn_oi) and not any(spikemon['t'][np.where(b_spike_from_nrn_oi)[0]] > min_t_first_spike_ms):
    #     conditions_not_met += [3]  # must_spike_late
    # if n_spikes_nrn_oi > max_n_spikes:
    #     conditions_not_met += [4]  # nrn_oi_few_spikes
    # if baseline_pm > -45:
    #     conditions_not_met += [5]  # baseline_low
    # if baseline_pm < -65:
    #     conditions_not_met += [6]  # baseline_high
    # if spikerate_int > 100:
    #     conditions_not_met += [7]  # IN_rate_low
    # if spikerate_int < 5:
    #     conditions_not_met += [8]  # IN_rate_high
    # if baseline_std_pm > 1.5:
    #     conditions_not_met += [9]  # baseline_stdev_high
    return conditions_not_met


def unwrap_arguments(argument_list):
    """Unwraps a list of arguments and calls run_simulation with the individual arguments contained in the list

    :param argument_list: list of arguments to be passed to function
    :type argument_list: list
    :return: returns a list of the return value(s) of run_simulation (if any), or an exception if one got raised
    """

    assert isinstance(argument_list, list) or isinstance(argument_list, tuple), "argument_list must be a list or tuple"

    try:
        rvalue1, rvalue2 = run_simulation(*argument_list)
    except Exception as e:
        rvalue1 = repr(e)
        rvalue2 = [-1]

    return [rvalue1, rvalue2]


def run_multiprocess(model_name, configs_list, filename_out_list, log_name=None, run_id=0, sim_time_ms=None,
                     clock_dt_ms=0.02, input_spiketimes_ms=None, b_verbose=False, batch_size=None,
                     recorded_variables='v', recording_start_ms=0, b_save_all=False):
    """Run multiple simulations in parallel. Takes a list of configs and ouput filenames and runs simulations based on
    each config in parallel using multiprocessing.

    :param model_name: filename of parameter config file (ending .json can be omitted from string). config will not be
        loaded from this file, but from the configs in configs_list below.
    :type model_name: str
    :param configs_list: list of config dictionaries, one dict per simulation to be run in parallel
    :type configs_list: list
    :param filename_out_list: list of filenames for the output .pkl files.
    :type filename_out_list: list
    :param log_name: see run_simulation() in this module
    :type log_name: str or None
    :param run_id: see run_simulation() in this module
    :type run_id: int
    :param sim_time_ms: see run_simulation() in this module
    :type sim_time_ms: int or None
    :param clock_dt_ms: see run_simulation() in this module
    :type clock_dt_ms: float
    :param input_spiketimes_ms: see run_simulation() in this module
    :type input_spiketimes_ms: [[[float]]] or None
    :param b_verbose: see run_simulation() in this module
    :type b_verbose: bool or int
    :param batch_size: number of processes (simulations) to run in a batch. None: no batching
    :type batch_size: int or None
    :param recorded_variables: (tuple of) name(s) of the variable(s) to be recorded in StateMonitor and saved to .pkl
    :type recorded_variables: str or tuple
    :param recording_start_ms: [default=0] start recording spikes and states at this timepoint into the simulation
    :type recording_start_ms: float or int
    :param b_save_all: if True, saving_conditions will be ignored, i.e. an ouput .pkl file saved for every sim run.
    :type b_save_all: bool or int
    :return:
    """

    # check inputs
    assert isinstance(configs_list, list) and \
           all([isinstance(configs_list[v], dict) for v in range(len(configs_list))]), \
           "run_multiprocess(): argument 'configs_list' must be a list of configuration dictionaries"
    assert isinstance(filename_out_list, list) and \
           all([isinstance(filename_out_list[v], str) for v in range(len(filename_out_list))]), \
           "run_multiprocess(): argument 'configs_list' must be a list of configuration dictionaries"
    assert len(configs_list) == len(filename_out_list), "configs_list and filename_out_list must be of same length"

    # get logger
    log = logging.getLogger(log_name)

    # Tell brian2 to not delete delete source files after compiling. The Cython source files can take a significant
    # amount of disk space, and are not used anymore when the compiled library file exists. However, parallel
    # processing, deleting source files while another process looks for them can be problematic,
    # see https://github.com/brian-team/brian2/issues/1081
    # b2.prefs.codegen.runtime.cython.delete_source_files = False
    # => didn't work, instead set cache size to 0 in file brian_preferences

    # to avoid memory issues, create a new pool every 100 simulations
    n_simulations = len(configs_list)
    n_sims_run = 0
    n_sims_saved = 0
    if batch_size:
        n_batches = np.ceil(n_simulations / batch_size).astype(int)
        log.info("_ starting run " + str(run_id) + " multiprocess (" + str(n_simulations) + " sims) with batch size " +
                 str(batch_size) + " ...")
    else:
        n_batches = 1
        log.info("_ starting run " + str(run_id) + " multiprocess (" + str(n_simulations) + " sims), no batches ...")
    log.debug('Simulations completed (/exception):')
    for i_batch in range(n_batches):
        # create context with start method "spawn" - see https://pythonspeed.com/articles/python-multiprocessing/
        # and https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods
        # NOTE: The 'spawn' and 'forkserver' start methods cannot currently be used with “frozen” executables (i.e.,
        # binaries produced by packages like PyInstaller and cx_Freeze) on Unix. The 'fork' start method does work.
        context = multiprocessing.get_context('forkserver')
        # create multiprocessing pool
        with context.Pool(np.round(os.cpu_count() * 3 / 4).astype(int)) as pool:
            if batch_size:
                sim_slice_cur_batch = slice(batch_size * i_batch, batch_size * i_batch + batch_size)
            else:
                sim_slice_cur_batch = slice(None)
            # run simulations
            for i, r in enumerate(pool.imap_unordered(unwrap_arguments, list(zip(itertools.repeat(model_name),
                                                                                 configs_list[sim_slice_cur_batch],
                                                                                 itertools.repeat(None),
                                                                                 itertools.repeat(run_id),
                                                                                 itertools.repeat(sim_time_ms),
                                                                                 itertools.repeat(clock_dt_ms),
                                                                                 itertools.repeat(input_spiketimes_ms),
                                                                                 filename_out_list[sim_slice_cur_batch],
                                                                                 itertools.repeat(recorded_variables),
                                                                                 itertools.repeat(recording_start_ms),
                                                                                 itertools.repeat(b_save_all),
                                                                                 itertools.repeat(b_verbose))))):
                if r[0].endswith('.pkl'):
                    # output id (end of filename) of completed sim to log file only (level debug does not go to console)
                    if any(r[1]):
                        log.debug(str(int(r[0][-10:-4])) + '\t' + str(r[1]))
                    else:
                        log.debug(str(int(r[0][-10:-4])))
                        n_sims_saved += 1
                else:
                    log.debug('exception: ' + r[0])
                # start timer once and calculte estimated time left every 100 sims
                if n_sims_run == 0:
                    start = timer()
                    last_timer = start
                    eta = 0
                n_sims_run += 1
                if n_sims_run % 100 == 0:
                    t_elapsed = timer() - last_timer
                    last_timer = timer()
                    eta = t_elapsed / 100 * (len(configs_list) - n_sims_run)
                # print to console number of sims done, percentage done and estimated time left
                print('\b' * 34 + '{0: >6} {1: >6}% {2: >6}min {3: >6}out'.format(str(n_sims_run),
                                                                                  str(np.round(100 * n_sims_run /
                                                                                        len(configs_list), decimals=1)),
                                                                                  str(round(eta / 60)),
                                                                                  str(n_sims_saved)),
                      end='', flush=True)
                # alternatively use starmap (doesn't return iteratively, so no progress bar possible)
                # pool.starmap(run_simulation, zip(itertools.repeat(model_name), ...))
            duration = timer() - start
            print('\b\b\b\b\b\b\b\b\b\b\b\b\b\b')
            if batch_size:
                log.info('batch {0: >3} done. runtime: {1: >3}s. estimated time left: {2} min'.format(str(i_batch),
                    str(round(duration, 1)), str(round(duration / batch_size * (n_simulations - n_sims_run) / 60, 1))))
    log.info('done')
