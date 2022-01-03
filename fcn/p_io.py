"""Functions for writing/reading data and figures to/from the hard drive, as well as logging. [call-time-model]
"""

import os
import sys
import json
import time  # used for datetime string in main log file
import copy
import socket  # used to get computer name for main log file
import shutil
import pickle
import logging
import scipy.io
import itertools
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from fcn import p_util

# "global" variables
BASE_PATH_CODE = ''
BASE_PATH_OUTPUT = ''  # change this to an absolute path to save output data (.pkl files) and figures there


def create_path_to_file(filename):
    """Check whether all directories in the path to the file exist and if not, create them.

    :param filename: filename (full path) to be checked
    :type filename: str
    :return:
    """

    split_filename = filename.split(os.path.sep)
    path_to_file = os.path.sep.join(split_filename[:-1])
    if path_to_file and not os.path.exists(path_to_file):
        os.makedirs(path_to_file, exist_ok=True)  # todo (for compatibility with Python < 3.2): remove "exist_ok=True"
        print("o created directories in the path to file (" + path_to_file + ") that did not exist before.")


def get_file_list(pathname, file_ending=''):
    """Get a (sorted) list of all filenames in a directory.

    :param pathname: full path of directory containing the files whose names are to be returned
    :type pathname: str
    :param file_ending: returned list will only contain filenames ending on this. can be a string or a list of strings,
        e.g. ['.png', '.jpg']
    :type file_ending: str or list
    :return: file_list: list of filenames
    :rtype: file_list: list
    """

    # if file_ending contains a single entry, convert it to a list
    if type(file_ending) is not list:
        file_ending = [file_ending]

    # check inputs
    assert os.path.isdir(pathname), 'ERROR: directory ' + pathname + ' was not found.'
    assert all(isinstance(item, str) for item in file_ending), "file_ending must be a string or list of strings"

    # normalize path string into proper OS string
    pathname = os.path.normpath(pathname)

    # get list of filenames
    file_list = [f for f in os.listdir(pathname)
                 if os.path.isfile(os.path.join(pathname, f)) and any([f.endswith(e) for e in file_ending])]
    file_list.sort()

    return file_list


def get_ids_from_file_list(file_list):
    """Takes a list of filenames with the format *_x.*, where x is a number, and returns a list of all x as integers.

    :param file_list: list of filenames in the format *_x.*
    :type file_list: list
    :return: ids: list of integers of the ids found in the file_list
    :rtype: ids: list
    """

    # check inputs
    assert isinstance(file_list, list) and all([isinstance(f, str) for f in file_list]), \
        "argument file_list must be a list of strings (filenames with the format *_x.*, where x is a number)"

    # loop through files and extract ids
    ids = []
    for filename in file_list:
        _, filename_no_ending = get_filename_from_path(filename, b_keep_ending=False)
        ids.append(int(filename_no_ending.split('_')[-1]))

    return ids


def get_filename_from_path(full_path, b_keep_ending=True):
    """Splits full path string (e.g. "/path/to/filename.end") and returns the file name ("filename.end" or "filename").

    :param full_path: full path to file
    :type full_path: str
    :param b_keep_ending: if False, remove file ending (everything including and following the last '.' in the filename)
    :type b_keep_ending: bool
    :return:    - pathname: pathname without filename
                - filename: filename with or without ending
    :rtype:     - pathname: str
                - filename: str
    """

    # check inputs
    assert isinstance(full_path, str), "argument 'full_path' must be a string."
    assert not full_path.endswith(os.path.sep), "filename seems to be missing from full_path: " + full_path

    # normalize path string into proper OS string
    full_path = os.path.normpath(full_path)

    # split path by path seperator
    split_path = full_path.split(os.path.sep)
    pathname = os.path.sep.join(split_path[0:-1])
    filename_with_ending = split_path[-1]

    # remove file ending
    if b_keep_ending or '.' not in filename_with_ending:
        filename = filename_with_ending
    else:
        filename = '.'.join(filename_with_ending.split('.')[0:-1])

    return pathname, filename


def load_config(model_name):
    """Load model parameters from config file (.json)

    :param model_name: filename of parameter config file (ending .json can be omitted from string). First checks if
        model_name is the filename (full path and ending on .json) of an existing file. filename_out will be None in
        that case. If not, it looks for the file in the default cfg directory.
    :type model_name: str
    :return:
        - config: dictionary of model parameters, as loaded from .json configuration file
        - filename_cfg: filename (full path) of the config file, including .json extension
        - filename_out; filename of the pickle file to be generated for simulation output. None, if full path was passed
    :rtype:
        - config: str
        - filename_cfg: str
        - filename_out: str or None
    """

    # check inputs
    assert isinstance(model_name, str), "argument 'model_name' must be a string"

    # filenames for config file to load (.json) and pickle file for saved results (.pkl)
    if model_name[-5:] == '.json':
        if os.path.isfile(model_name):
            filename_cfg = model_name
            filename_out = None
        else:
            filename_cfg = BASE_PATH_CODE + 'cfg' + os.path.sep + model_name
            filename_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + model_name[0:-5] + '.pkl'
    else:
        filename_cfg = BASE_PATH_CODE + 'cfg' + os.path.sep + model_name + '.json'
        filename_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + model_name + '.pkl'

    # load model config file (.json)
    try:
        with open(filename_cfg) as f:
            config = json.load(f)
    except FileNotFoundError:
        print('ERROR: ' + filename_cfg + ' was not found.')
        raise
    except json.decoder.JSONDecodeError:
        print('ERROR: ' + filename_cfg + ' was found but could not be read. Is it a properly formatted .json file?')
        raise

    return config, filename_cfg, filename_out


def check_config(config, model_name):
    """Perform some checks on the contents of the model configuration file (e.g. are number of parameter values
    consistent with the number of populations/synapses, etc.)

    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :param model_name: filename of .json config file (or model name, i.e. filename without extension)
    :type model_name: str
    :return:
    """

    n_pops = len(config['misc']['population_id'])
    n_syns = len(config['synapse']['syn_pre_idx'])
    free_parameter_keys = list(config['free_parameter_stepsize'].keys())

#    assert len(config['misc']['n_neurons_per_pop']) == len(config['misc']['population_id']), \
#        model_name + ": n_neurons_per_pop must have same number of elements as population_id"

    if 'wnoise_cv' in config['input_current']:
        assert 'wnoise_dt' in config['input_current'], \
            model_name + ": if wnoise_cv in input_current, wnoise_dt is also needed"
        assert config['input_current']['wnoise_dt'] > 0, \
            model_name + ": wnoise_dt in input_current must be a positive integer or float"

    for p, val in config['parameters_nrn'].items():
        if type(val[0]) is list:
            assert p in free_parameter_keys, model_name + ": parameter '" + p + "' in 'parameters_nrn' contains a" + \
                " list of lists, but is not a free parameter (i.e. listed in 'free_parameters_stepsize')"
            assert len(val) == 2 and [len(val[i]) == n_pops for i in range(2)], \
                model_name + ": free parameter '" + p + "' in 'parameters_nrn' must be a list of two " + \
                "lists (lower and upper bounds of parameter range) with each list containing " + str(n_pops) + \
                " values (number of populations, i.e. elements in misc['population_id'])"
        else:
            assert len(val) == n_pops, model_name + ": parameter '" + p + "' in 'parameters_nrn' must be list of " + \
                str(n_pops) + " values (number of populations, i.e. elements in misc['population_id'])."

    for p, val in config['parameters_syn'].items():
        if val and type(val[0]) is list:
            assert p in free_parameter_keys, model_name + ": parameter '" + p + "' in 'parameters_syn' contains a" + \
                " list of lists, but is not a free parameter (i.e. listed in 'free_parameters_stepsize')"
            assert len(val) == 2 and [len(val[i]) == n_syns for i in range(2)], \
                model_name + ": free parameter '" + p + "' in 'parameters_syn' must be a list of two " + \
                "lists (lower and upper bounds of parameter range) with each list containing " + str(n_syns) + \
                " values (number of synapses, i.e. elements in synapse['syn_pre_idx'])"
        else:
            assert len(val) == n_syns, model_name + ": parameter '" + p + "' in 'parameters_syn' must be list of " + \
                str(n_syns) + " values (number of synapses, i.e. elements in synapse['syn_pre_idx'])."

    for k in free_parameter_keys:
        if k in config['parameters_nrn']:
            assert type(config['parameters_nrn'][k]) is list and len(config['parameters_nrn'][k]) == 2, model_name + \
                ": parameter '" + k + "' in 'parameters_nrn' listed as free parameter (in 'free_parameter_stepsize" + \
                "') and must therefore be a list of two lists (lower and upper bounds of parameter range)"
        elif k in config['parameters_syn']:
            assert type(config['parameters_syn'][k]) is list and len(config['parameters_syn'][k]) == 2, model_name + \
                ": parameter '" + k + "' in 'parameters_syn' listed as free parameter (in 'free_parameter_stepsize" + \
                "') and must therefore be a list of two lists (lower and upper bounds of parameter range)"
            assert all([config['parameters_syn'][k][1][i] >= config['parameters_syn'][k][0][i]
                        for i in range(len(config['parameters_syn'][k][0]))]), \
                "second free parameter value cannot be smaller than first"
        elif 'parameters_gen' in config and k in config['parameters_gen']:
            assert type(config['parameters_gen'][k]) is list and len(config['parameters_gen'][k]) == 2, model_name + \
                ": parameter '" + k + "' in 'parameters_gen' listed as free parameter (in 'free_parameter_stepsize" + \
                "') and must therefore be a list of two lists (lower and upper bounds of parameter range)"
        elif k in config['input_current'] and k == 'offset_range':
            assert type(config['input_current'][k]) is list and len(config['input_current'][k]) == 2, model_name + \
                ": parameter '" + k + "' in 'input_current' listed as free parameter (in 'free_parameter_stepsize" + \
                "') and must therefore be a list of two lists (lower and upper bounds of parameter range)"
        elif k in config['input_current']:
            assert type(config['input_current'][k]) is list and len(config['input_current'][k]) == 2 and \
                type(config['input_current'][k][0] is list), model_name + ": parameter '" + k + "' in " + \
                "'input_current' listed as free parameter (in 'free_parameter_stepsize') and must therefore be a " + \
                "list of (two) lists (lower and upper bounds of parameter range) of list (currents of respective pop)"


def reformat_config(filename, b_create_backup=True):
    """Reads an automatically generated .json config file (e.g. using p_macro.config()) and removes certain newlines and
    other whitespaces for better readability. overwrites the original .json file after (optionally) creating a backup.

    :param filename: filename (full path) of the .json config file to be reformatted.
    :type filename: str
    :param b_create_backup: crates a backup file in the same path before overwriting the original. appends
        '_unformatted' to the filename.
    :type b_create_backup: bool
    :return:
    """

    with open(filename) as file:
        newline_got_stripped_from_last_line = False
        out_lines = []
        for line in file:
            stripped = line.strip()  # removes whitespaces from beginning and end
            if stripped.endswith('[') and not stripped.startswith('"ode"'):
                out_lines.append(line.rstrip())
                newline_got_stripped_from_last_line = True
            elif stripped[0].isdigit() or stripped[0] == '-':
                out_lines.append(' ' + line.strip())
                newline_got_stripped_from_last_line = True
            else:
                if newline_got_stripped_from_last_line:
                    # remove whitespaces (tabs) at beginning because this line got merged with the previous line
                    out_lines.append(line.lstrip())
                else:
                    out_lines.append(line)
                newline_got_stripped_from_last_line = False

    if b_create_backup:
        shutil.copyfile(filename, os.path.splitext(filename)[0] + '_unformatted.json')

    with open(filename, 'w') as file:
        for line in out_lines:
            file.write(line)


def get_param_exploration_configs(config_lo, config_hi, b_generate_configs=True):
    """Takes two config files and returns a config file for each unique combination of parameter values, interpolated
    between the values of config_lo and config_hi, incremented by the values in config_hi['step_size'] for each param.

    :param config_lo: regular config file as loaded from .json, containing lower limits of all parameters
    :type config_lo: dict
    :param config_hi: config file containing the upper limits for only those parameters that should be varied. Needs to
        contain an additional dict 'step_size' that has names of all explored parameters as keys and as values the
        amount by which the parameter values are to be interpolated between those in config_lo and config_hi
    :type config_hi: dict
    :param b_generate_configs: set False if 'configs' and 'log_lines' are not actually needed. both will be an
        empty list in that case
    :type b_generate_configs: bool
    :return:    - configs: list of config dictionaries
                - log_lines: list of one line of text for each config with explored parameter values for log file
                - combinations: list of unique combinations of values for the explored parameters (len==len(configs))
                - combo_param_names: list of names of all explored parameters, each repeated, so that
                    combo_param_names[i] is the name of the parameter whose value is at combinations[sim][i]
                - all_values_per_param: list of lists, each sublist containing all unique values of the respective
                    parameter. len==len(combo_param_names)
    :rtype: - configs: list
            - log_lines: list
            - combinations: list
            - combo_param_names: list
            - all_values_per_param: list
    """

    # check inputs
    assert isinstance(config_lo, dict) and isinstance(config_hi, dict), 'config_lo and config_hi must both be dicts'
    assert 'step_size' in config_hi, "config_hi must contain a key 'step_size' with stepsizes for all explored params"

    # get names of top level entries (sub-dicts) in config_hi, i.e. groups that contain parameters to be explored
    param_dict_names = list(config_hi.keys())
    param_dict_names.remove('step_size')

    # for each top level dict, get names and number of values of all parameters to be explored
    param_names = []
    n_values_per_param = []
    all_values_per_param = []
    combo_param_names = []  # will hold the parameter name for each explored value (value in sublist of combinations)
    for i_dict, dict_name in enumerate(param_dict_names):
        param_names.append(list(config_hi[dict_name].keys()))
        n_values_per_param.append([])
        for par in param_names[i_dict]:
            if isinstance(config_hi[dict_name][par], list):
                n_values_per_param[i_dict].append(len(config_lo[dict_name][par]))
                param_values_lo = config_lo[dict_name][par]
                param_values_hi = config_hi[dict_name][par]
                step_sizes = config_hi['step_size'][par]
            else:
                n_values_per_param[i_dict].append(1)
                param_values_lo = [config_lo[dict_name][par]]
                param_values_hi = [config_hi[dict_name][par]]
                step_sizes = [config_hi['step_size'][par]]

            # loop through all values and create a list of all (interpolated) values between lower and upper limit
            for i_val in range(len(param_values_lo)):
                combo_param_names.append(par)
                val_lo = param_values_lo[i_val]
                val_hi = param_values_hi[i_val]
                difference = val_hi - val_lo
                stepsize = step_sizes[i_val]
                if stepsize > 0:
                    if par == 'n_neurons_per_pop':
                        all_values_per_param.append(np.linspace(val_lo, val_hi,
                                                                round(difference / stepsize) + 1).astype(int).tolist())
                    else:
                        all_values_per_param.append(np.linspace(val_lo, val_hi,
                                                                round(difference / stepsize) + 1).tolist())
                else:
                    all_values_per_param.append([val_lo])

    # generate all unique combinations of values for the explored parameters
    combinations = list(itertools.product(*all_values_per_param))

    # loop through all combinations and generate a config dict with the respective values
    configs = []
    log_lines = []
    if b_generate_configs:
        for i_conf, combo in enumerate(combinations):
            configs.append(copy.deepcopy(config_lo))
            log_lines.append(str(i_conf).zfill(6) + '\t{')
            val_count = 0
            for i_dict, dict_name in enumerate(param_dict_names):
                log_lines[i_conf] += dict_name + ':'
                for par in param_names[i_dict]:
                    if isinstance(configs[i_conf][dict_name][par], list):
                        param_values = configs[i_conf][dict_name][par]
                    else:
                        param_values = [configs[i_conf][dict_name][par]]
                    log_lines[i_conf] += par + ':['
                    for i_val in range(len(param_values)):
                        log_lines[i_conf] += str(np.around(combo[val_count], decimals=6)) + ','
                        if isinstance(configs[i_conf][dict_name][par], list):
                            configs[i_conf][dict_name][par][i_val] = combo[val_count]
                        else:
                            configs[i_conf][dict_name][par] = combo[val_count]
                        val_count += 1
                    log_lines[i_conf] += ']\t'
            log_lines[i_conf] += '\b\n'

    return configs, log_lines, combinations, combo_param_names, all_values_per_param


def save_monitors(statemons, spikemons, spikes_poisson, connectivity, connectivity_generator, config, info, filename):
    """Save states of brian2 StateMonitors and SpikeMonitors to binary file, using pickle. (Load with load_monitors())

    :param statemons: dictionary or list of dictionaries returned by brian2.StateMonitor.get_states()
    :type statemons: dict or list
    :param spikemons: dictionary or list of dictionaries returned by brian2.SpikeMonitor.get_states()
    :type spikemons: dict or list
    :param spikes_poisson: dict returned by b2.SpikeMonitor.get_states() containing spikes of poisson_group or None
    :type spikes_poisson: dict or None
    :param connectivity: dictionary or list of dictionaries containing synaptic connectivity information
    :type connectivity: dict or list
    :param connectivity_generator: dictionary or list of dictionaries containing spikegenerator connectivity information
    :type connectivity_generator: dict or list
    :param config: dictionary of model parameters, as loaded from .json configuration file
    :type config: dict
    :param info: dictionary or list of dictionaries containing information about simulation, created by run_simulation()
    :type info: dict or list
    :param filename: filename (full path) of output file
    :type filename: str
    :return: log_message: success message for logging
    :rtype: log_message: str
    """

    # if statemons and spikemons contain a single object (dictionary), convert them to lists
    if type(statemons) is not list:
        statemons = [statemons]
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(connectivity) is not list:
        connectivity = [connectivity]
    if type(connectivity_generator) is not list:
        connectivity_generator = [connectivity_generator]
    if type(info) is not list:
        info = [info]

    # check inputs
    assert all(isinstance(item, dict) for item in statemons), \
        "statemons must be a dictionary or list of dictionaries returned by brian2.SpikeMonitor.get_states()"
    assert all(isinstance(item, dict) for item in spikemons), \
        "spikemons must be a dictionary or list of dictionaries returned by brian2.SpikeMonitor.get_states()"
    assert all(isinstance(item, dict) for item in connectivity), \
        "connectivity must be a dictionary or list of dictionaries"
    assert all(isinstance(item, dict) for item in info), \
        "info must be a dictionary or list of dictionaries"
    assert len(statemons) is len(spikemons) is len(info), \
        "statemons, spikemons and info must all have the same number of elements"

    # check whether all directories in the path to the file exist and if not, create them
    create_path_to_file(filename)

    # open output file
    with open(filename, 'wb') as file_out:
        # dump data to file
        pickle.dump([statemons, spikemons, spikes_poisson, connectivity, connectivity_generator, config, info],
                    file_out)

    return "o monitor data and info saved as " + filename


def load_monitors(filename, b_silent=False, b_return_first_element_only=False):
    """Load states of brian2 StateMonitors and SpikeMonitors from binary file, using pickle. (Save with save_monitors())
    Dictionaries containing the states will be converted to SimpleNamespaces and returned as statemons and spikemons.

    :param filename: filename (full path) of input file created by p_io.save_monitors(), containing brian2 monitor data
    :type filename: str
    :param b_silent: if True, don't print success
    :type b_silent: bool
    :param b_return_first_element_only: if True, return not the whole lists, but only the first element of each list
    :type b_return_first_element_only: bool
    :return:
        - statemons: list of brian2 StateMonitor - like(!) SimpleNamespaces retrieved from the file
        - spikemons: list of brian2 SpikeMonitor - like(!) SimpleNamespaces retrieved from the file
        - spikemon_poisson: brian2 SpikeMonitor - like(!) SimpleNamespace containing spikes of poisson_group or None
        - connectivity: list of dictionaries containing synaptic connectivity information
        - connectivity_generator: list of dictionaries containing spikegenerator connectivity information (can be empty)
        - config: dictionary as loaded from .json configuration file
        - info: list of dictionaries containing additional information about simulation, created by run_simulation()
    :rtype:
        - statemons: [SimpleNamespace]
        - spikemons: [SimpleNamespace]
        - spikemon_poisson: SimpleNamespace or None
        - connectivity: [dict]
        - connectivity_generator: [dict]
        - config: dict
        - info: [dict]
    """

    # open input file
    with open(filename, 'rb') as file_in:
        # load dictionaries containing monitor states from file
        state_dicts, spike_dicts, spike_dict_poisson, connectivity, connectivity_generator, config, info \
            = pickle.load(file_in)

    statemons = []
    spikemons = []

    # loop through dictionaries and convert data structure to something similar to brian2 Monitor objects
    for statemon, spikemon, in zip(state_dicts, spike_dicts):
        for el in statemon:
            if len(statemon[el].shape) > 1:
                statemon[el] = np.swapaxes(statemon[el], 0, 1)
        for el in spikemon:
            if len(spikemon[el].shape) > 1:
                spikemon[el] = np.swapaxes(spikemon[el], 0, 1)

        statemons.append(SimpleNamespace(**statemon))
        spikemons.append(SimpleNamespace(**spikemon))

    if spike_dict_poisson:
        spikemon_poisson = SimpleNamespace(**spike_dict_poisson)
    else:
        spikemon_poisson = None

    if not b_silent:
        print("i monitor data and info loaded from " + filename)
        if b_return_first_element_only:
            print('-> WARNING: load_monitors() only returned the first element of each list (e.g. statemon)')

    if b_return_first_element_only:
        if not connectivity_generator:
            return statemons[0], spikemons[0], spikemon_poisson, connectivity[0], connectivity_generator, config, \
                   info[0]
        else:
            return statemons[0], spikemons[0], spikemon_poisson, connectivity[0], connectivity_generator[0], config, \
                   info[0]
    else:
        return statemons, spikemons, spikemon_poisson, connectivity, connectivity_generator, config, info


def aggregate_monitors_from_parex(pathname, idx_nrn_to_subset_abs=None):
    """Collect output data from a parameter exploration (all .pkl files in directory) and aggregate all state monitors,
    spike monitors, info dicts, etc. into a multi-run list for each, for further analysis. Returns are similar to
    load_monitors(), but each element in the lists (e.g. spikemons) is the ouput of a different parex simulation.
    Optionally return only the data of a single neuron to save memory, given by idx_nrn_to_subset_abs.

    :param pathname: full PATH to directory containing .pkl output files of a parameter exploration run
    :type pathname: str
    :param idx_nrn_to_subset_abs: absolute index to the neuron for which spike- and state-data should be extracted and
        returned. if None, whole monitors will be returned (potentially very memory intensive - avoid if not needed)
    :type idx_nrn_to_subset_abs: int or None
    :return:
        - statemons: list of brian2 StateMonitor - like(!) SimpleNamespaces aggregated from the files in pathname
        - spikemons: list of brian2 SpikeMonitor - like(!) SimpleNamespaces aggregated from the files
        - spikemon_poisson: list of brian2 SpikeMonitor-like SimpleNamespace containing spikes of poisson_group or Nones
        - connectivity: list of dictionaries containing synaptic connectivity information
        - connectivity_generator: list of dictionaries containing spikegenerator connectivity information (can be empty)
        - config: list of dictionaries as loaded from .json configuration file
        - info: list of dictionaries containing additional information about simulation, created by run_simulation()
    :rtype:
        - statemons: [SimpleNamespace]
        - spikemons: [SimpleNamespace]
        - spikemon_poisson: [SimpleNamespace] or [None]
        - connectivity: [dict]
        - connectivity_generator: [dict]
        - config: [dict]
        - info: [dict]
    """

    # get list of .pkl files
    file_list = get_file_list(pathname, '.pkl')
    pathname = os.path.normpath(pathname) + os.path.sep

    # declare variables as empty lists
    statemons, spikemons, spikemon_poisson, connectivity, connectivity_generator, config, info \
        = ([None] * len(file_list) for _ in range(7))

    for i, filename in enumerate(file_list):
        statemon_c, spikemon_c, spikemon_poisson[i], connectivity[i], connectivity_generator[i], config[i], info_c \
            = load_monitors(os.path.sep.join((pathname, filename)), b_silent=True, b_return_first_element_only=True)
        if idx_nrn_to_subset_abs is not None:
            statemon_sub, spikemon_sub, info_sub = p_util.subset_neuron_from_monitors([statemon_c], [spikemon_c],
                                                                                      [info_c], idx_nrn_to_subset_abs)
            statemons[i] = statemon_sub[0]
            spikemons[i] = spikemon_sub[0]
            info[i] = info_sub[0]
        else:
            statemons[i] = statemon_c
            spikemons[i] = spikemon_c
            info[i] = info_c

    return statemons, spikemons, spikemon_poisson, connectivity, connectivity_generator, config, info


def save_figures(figure_handles, base_filename, figure_format='png', dpi=300, box_mrgns='tight', run_id=-999,
                 b_close_figures=False, b_watermark=True):
    """Saves figures identified by their figure handles in a format supported by the active backend for pyplot,
    e.g. png, pdf, ...

    :param figure_handles: handle or list of handles to the figure(s) to be plotted
    :type figure_handles: matplotlib.figure.Figure or list
    :param base_filename: filename (full path) of the figure (without extension). If multiple handles are passed,
        base_filename is appended with a running number (three digits, zero-padded)
    :type base_filename: str
    :param figure_format: [default='png'] format supported by the active backend for pyplot, in which the figure is to
        be saved (e.g. 'png' or 'pdf'). This will also serve as the filename extension. Can be a list of formats.
    :type figure_format: str or list
    :param dpi: [default=300] dots per inch of saved figure. None: use system default
    :type dpi: int or float or None
    :param box_mrgns: [default='tight'] set to None to disable "tight" bbox (useful for pixel-perfect conn matrix)
    :type box_mrgns: str or None
    :param run_id: integer identifier for the simulation run this figure was produced from. gets embedded as metadata
        under key 'Title'. Default if no run_id was passed: -999.
    :type run_id: int
    :param b_close_figures: [default=False] if true, figure windows will be closed after saving
    :type b_close_figures: bool
    :param b_watermark: [default=False] if true, plot the run_id as a watermark in the bottom right corner of the figure
    :type b_watermark: bool
    :return:
    """

    # if figure_handles contains a single handle, convert it to a list
    if type(figure_handles) is not list:
        figure_handles = [figure_handles]

    # if figure_format contains a single string, convert it to a list
    if type(figure_format) is not list:
        figure_format = [figure_format]

    # check whether all directories in the path to the file exist and if not, create them
    create_path_to_file(base_filename)

    # save figure(s)
    for i_format in range(len(figure_format)):
        if len(figure_handles) == 1:
            if b_watermark and run_id > 0:
                figure_handles[0].text(.9, .1, str(run_id), ha='right', va='bottom', fontsize=6,
                                       transform=figure_handles[0].transFigure, color=[.7, .7, .7])
            figure_handles[0].savefig(base_filename + '.' + figure_format[i_format],
                                      format=figure_format[i_format], bbox_inches=box_mrgns, dpi=dpi,
                                      metadata={'Title': str(run_id).zfill(4)})
            print("o figure saved as " + base_filename + '.' + figure_format[i_format])
            if b_close_figures:
                plt.close(figure_handles[0])
        else:
            for f in range(len(figure_handles)):
                if b_watermark and run_id > 0:
                    figure_handles[f].text(.9, .1, str(run_id), ha='right', va='bottom', fontsize=6,
                                           transform=figure_handles[f].transFigure, color=[.7, .7, .7])
                figure_handles[f].savefig(base_filename + '_' + str(f).zfill(3) + '.' + figure_format[i_format],
                                          format=figure_format[i_format], bbox_inches=box_mrgns, dpi=dpi,
                                          metadata={'Title': str(run_id).zfill(4)})
                if b_close_figures:
                    plt.close(figure_handles[f])
            print("o figures saved as " + base_filename + '_***.' + figure_format[i_format])


def load_traces_from_mat(filename):
    """Load voltage traces from .mat file (electroGUI format)

    :param filename: filename of a matlab .mat file containing traces (cell array "traces" with one cell per trial)
    :type filename: str
    :return: traces: list of numpy.arrays of voltage traces (each array corresponds to a "traces" cell, i.e. trial)
    :rtype: [np.ndarray]
    """

    # load event stream matrix from .mat file
    recording_mat = scipy.io.loadmat(filename)

    traces = [np.squeeze(np.squeeze(recording_mat['traces'])[trc]) for trc in range(recording_mat['traces'].size)]

    print("i traces loaded from " + filename)

    return traces


def load_neuronexus_spiketimes_from_mat(filename, neuron_ids=None):
    """Load spiketimes of extracellular recordings aligned to playback onset in seconds from Matlab .mat file. file
    must contain three variables: spktms - cell array of unit-playback combinations, each cell containing one array of
    aligned spiketimes for each trial; info_bslcup - cell array containing an array of indices for each unit-playback
    combination of (in order): bird, location, session, channel, unit, playback; playback_name - cell array containing
    the name of the playback file for each unit.

    :param filename: filename (full path) of a matlab .mat file containing spiketimes (e.g. 'spks_rfrmt.mat')
    :type filename: str
    :param neuron_ids: list of integer ids, one for each unit in the .mat file. if None, use sequential numbers for ids.
    :type neuron_ids: list
    :return:    - spiketimes: array of aligned spiketimes of each unit-playback combination, one sub-array per trial
                - info_bslcup: array of indices len(spiketimes) of: Bird, Session, Location, Channel, Unit, Playback
                - playback_name: array of name of playback file for each unit-playback combination, i.e. len(spiketimes)
    :rtype:     - spiketimes: numpy.ndarray
                - info_bslcup: numpy.ndarray
                - playback_name: numpy.ndarray
    """

    # get data from .mat file
    spiketimes_mat = scipy.io.loadmat(filename)
    spiketimes = spiketimes_mat['spktms'][0]
    playback_name = spiketimes_mat['playback_type'][0]
    playback_dur = spiketimes_mat['playback_dur'][0]
    assert 'info_bslcup' in spiketimes_mat or 'info_snp' in spiketimes_mat, \
        'neither info_bslcup nor info_snp found in ' + filename
    if 'info_bslcup' in spiketimes_mat:
        info = spiketimes_mat['info_bslcup'][0]
    elif 'info_snp' in spiketimes_mat:
        info = spiketimes_mat['info_snp'][0]

    # remove singleton dimensions
    for u in range(len(spiketimes)):
        spiketimes[u] = np.squeeze(spiketimes[u], 0)
        for t in range(len(spiketimes[u])):
            spiketimes[u][t] = np.squeeze(spiketimes[u][t], 1)
    for u in range(len(playback_dur)):
        playback_dur[u] = np.squeeze(playback_dur[u], 0)
        playback_name[u] = playback_name[u][0]
        info[u] = np.squeeze(info[u], 0)
    playback_dur = np.concatenate(playback_dur)

    # convert values from seconds to milliseconds
    spiketimes = spiketimes * 1000
    playback_dur = playback_dur * 1000

    # generate a sequential integer id for each unit (all unit-playback combinations from the same unit get the same id)
    bslcu = [info[i][0:-1] for i in range(len(info))]
    _, continuous_unit_idx = np.unique(bslcu, return_inverse=True, axis=0)

    # assign either the supplied neuron ids, or the sequential ids
    if neuron_ids is not None:
        unit_id = np.array(neuron_ids)[continuous_unit_idx]
    else:
        unit_id = continuous_unit_idx

    # categorize each playback into a type (call, noise, ...)
    playback_type = []
    call_names = ['d_a_call_norm', 'a_call_norm', '2102call-1st_m', '2102call-1st_15km']
    noise_names = ['b_noise_calllength_min5amp_refade', 'b_noise_calllength_min15db_refade',
                   'b_noise_calllength_min10db_m', 'b_noise_calllength_min25db_refade']
    bos_names = ['19_579_BOS_norm_fade', '19_579_BOS_norm_fade_m', 'c_bos461_song_norm', '19_401_BOS_15km']
    song_names = ['440_song_ampmin2_refade', '440_song_ampmin2_refade_m', 'a440_song_norm']
    marker_names = ['15khz_10ms_marker', '15khz_10ms_marker_catch']
    f4054_names = ['F4054_norm_fade', 'F4054_norm_fade_m']
    jam_names = ['2102call-1st_15km_230dub', '2102call-1st_15km_230dub_catch']
    for i, name in enumerate(playback_name):
        if name in call_names:
            playback_type.append('call')
        elif name in noise_names:
            playback_type.append('noise')
        elif name in bos_names:
            playback_type.append('bos')
        elif name in song_names:
            playback_type.append('song')
        elif name in marker_names:
            playback_type.append('marker')
        elif name in f4054_names:
            playback_type.append('f4054')
        elif name in jam_names:
            playback_type.append('jam')
        else:
            playback_type.append('unknown')

    return spiketimes, info, playback_name, playback_dur, playback_type, unit_id


def save_spiketimes_to_pkl(spiketimes_ms, filename):
    """Save list of lists of spike times (in milliseconds; one sublist per trace) to pickle file (.pkl). Use for spikes
    detected in recordings.

    :param spiketimes_ms: list of lists of spike times (in milliseconds; one sublist per trace)
    :type spiketimes_ms: [[float]]
    :param filename: filename (full path) of output .pkl file
    :type filename: str
    :return:
    """

    # check inputs
    assert type(spiketimes_ms) is list, \
        "spiketimes_ms must be a list of lists of spiketimes (in milliseconds; one sublist per trace)"
    assert all(isinstance(item, list) for item in spiketimes_ms), \
        "spiketimes_ms must be a list of lists of spiketimes (in milliseconds; one sublist per trace)"

    # check whether all directories in the path to the file exist and if not, create them
    create_path_to_file(filename)

    # open output file
    with open(filename, 'wb') as file_out:
        # dump data to file
        pickle.dump(spiketimes_ms, file_out)

    print("o spiketimes saved as " + filename)


def load_spiketimes_from_pkl(filename):
    """Load list of spiketimes (in milliseconds; one sublist per trace) from pickle (.pkl) file created by
    save_spiketimes_to_pkl(). Use for spikes detected in recordings. To load spiketimes from model results, use
    p_io.load_monitors() and p_util.get_spiktimes_from_monitor().

    :param filename: filename (full path) of .pkl file containing spiketimes
    :type filename: str
    :return: spiketimes_ms: list of lists of spike times (in milliseconds; one sublist per trace)
    :rtype: [[float]]
    """

    # open input file
    with open(filename, 'rb') as file_in:
        # load dictionaries containing monitor states from file
        spiketimes_ms = pickle.load(file_in)

    print("i spiketimes loaded from " + filename)

    return spiketimes_ms


def create_logger(logger_name, filename_log=None, b_log_to_console=True):
    """Create a logger for printing to console and/or log file, using "logging" from the python standard library

    :param logger_name: name of the logger object
    :type logger_name: str
    :param filename_log: [default=None] filename of log file (text file, full path). if None, only log to console.
        serial number will be appended.
    :type filename_log: str or None
    :param b_log_to_console: [default=True] set False to suppress ouput to console for this logger
    :type b_log_to_console: bool
    :return: log: reference to logger object
    :rtype: logging.Logger
    """

    # check inputs
    assert filename_log is None or isinstance(filename_log, str), "filename_log must either be None or a string"

    # create logger object
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    console_formatter = logging.Formatter('%(message)s')
    file_formatter = logging.Formatter('%(message)s')

    if b_log_to_console:
        # create console handler with log level DEBUG and up (i.e. all logs)
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        log.addHandler(console_handler)

    if filename_log:
        # check whether all directories in the path to the file exist and if not, create them
        create_path_to_file(filename_log)

        # create file handler with log level INFO and up (i.e. excluding DEBUG logs)
        file_handler = logging.FileHandler(filename_log)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)
        log.addHandler(file_handler)

    return log


def get_next_log_id(filename_main_log):
    """ONLY GETS CALLED INTERNALLY. Return the next free id for run log file. opens main log (filename_main_log) and
    returns the number (id) at the beginning of the last line plus one.

    :param filename_main_log: filename of main log file (text file, full path)
    :type filename_main_log: str
    :return: next_id: integer id for next log file
    :rtype: int
    """

    # check inputs
    assert isinstance(filename_main_log, str), "filename_main_log must be a string"

    # check if main log exist and is not empty
    try:
        file_in = open(filename_main_log, 'r')
    except FileNotFoundError:
        next_id = 1
        print("! main log file " + filename_main_log + " not found -> starting with log id 0001")
    else:
        if os.stat(filename_main_log).st_size == 0:
            next_id = 1
        else:
            # as the file exists and is not empty, get the id at the beginning of the last line
            lines = file_in.read().splitlines()
            last_line = lines[-1]
            try:
                last_id = int(last_line[0:4])
            except ValueError:
                raise ValueError("first characters of last line of main log file " + filename_main_log + ": " +
                                 last_line[0:4] + " don't translate to an integer id.")
            else:
                next_id = last_id + 1

    return next_id


def set_up_loggers(base_path_output, model_name, b_debugging=False, b_parex=False, b_query_comment=True):
    """Set up and return two logger instances: 'main' and 'run'. 'main' for ouputting one line to a log file for each
    run, containing run_id, date and time, name of computer it was run on, model config used, output path and commit
    hash. 'run' outputs run info to console and log file, as well as copying the contents of the config to log file.
    only the name of the 'run' logger will be returned. use it to get the instance via logging.getLogger(log_run_name).
    also returns the run_id.

    :param base_path_output: path to directory where log files will be created/modified in a subdirectory log/
    :type base_path_output: str
    :param model_name: name of the model configuration, i.e. .json filename (no path) without the file extension (.json)
    :type model_name: str
    :param b_debugging: [default=False] if True, add flag to current line in main log file indicating debugging run
    :type b_debugging: bool
    :param b_parex: [default=False] if True, add flag to current line in main log file indicating parameter exploration
    :type b_parex: bool
    :param b_query_comment: [default=True] if False, don't ask the user to input a comment string
    :type b_query_comment: bool
    :return:
        - log_run_name: name of the logger instance of the python logging package to be used to ouput info during run
        - run_id: integer id for tracking the simulation run in log files, etc.
    :rtype:
        - log_run_name: str
        - run_id: int
    """

    # name for the logger instance used to ouput information during run_simulation()
    log_main_name = 'main'
    log_run_name = 'run'

    # get next free log id by checking last line of the main log file (if it exists) using next_log_id in this module
    path_to_logfiles = base_path_output + 'log' + os.path.sep
    run_id = get_next_log_id(path_to_logfiles + '0000_main.log')

    # try to get the current git commit hash to add it to the main log file. allows to retrace on which version it ran
    try:
        cur_git_commit_hash = \
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).strip().decode(sys.stdout.encoding)
    except Exception:
        cur_git_commit_hash = 'could_not_get_hash'

    # create the two loggers
    log_main = create_logger(log_main_name, path_to_logfiles + '0000_main.log', b_log_to_console=False)
    log_run = create_logger(log_run_name, path_to_logfiles + str(run_id).zfill(4) + '_' + model_name + '.log')

    # add line to main logger and run logger for current run
    log_line = str(run_id).zfill(4) + '\t' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()) + '\t' + \
        socket.gethostname() + '\t' + model_name + '\t' + base_path_output + '\t' + cur_git_commit_hash
    if b_debugging:
        log_line = log_line + '\tDEBUG'
    if b_parex:
        log_line = log_line + '\tPAREX'

    # optionally let the user enter a comment (e.g. what was changed in config since last run)
    if b_query_comment:
        comment = input("Enter a short comment on the current run for log files (Press Enter to skip):")
        if comment:
            log_line = log_line + '\t' + comment

    log_main.info(log_line)
    log_run.info(log_line)

    print("o created logger - filename: " + path_to_logfiles + str(run_id).zfill(4) + '_' + model_name + '.log')

    # return the name for the run logger
    return log_run_name, run_id


def clear_loggers():
    """Clear the default loggers created with set_up_loggers() ('main' and 'run') for reuse by removing all handlers
    from the logger objects.
    """

    log_names = ['main', 'run']

    for log_name in log_names:
        logger = logging.getLogger(log_name)
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])

    return
