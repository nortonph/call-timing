"""Miscellaneous higher level functions called by run_all.py and fig_ms.py. [call-time-model]
"""

import os
import shutil
# import my functions
from mod import lif
from fcn import p_io, p_plot, p_analysis
from fcn.p_io import BASE_PATH_OUTPUT

FIGURE_FORMATS = ['png']  # can also be multiple formats, e.g.: ['pdf', 'png']; connection matrices "hardcoded" as png


def parameter_exploration(config_name_lo, config_name_hi, input_spiketimes_ms, b_verbose=False, b_debugging=False,
                          b_multiprocessing=False, batch_size=None, recorded_variables='v', recording_start_ms=0,
                          b_save_all=False, b_query_log_comment=True, b_run_id_in_filename=True):
    """Runs multiple simulations over a range of parameter values, defined by two config files. Parameter values are
    interpolated between the values of config_lo and config_hi, incremented by the values in config_hi['step_size']
    for each param.

    :param config_name_lo: filename of regular .json config file, containing lower limits of all parameters
    :type config_name_lo: str
    :param config_name_hi: filename of config .json file containing the upeer limits for only those parameters that
        should be varied. Needs to contain an additional dict 'step_size' that has names of all explored parameters as
        keys and as values the amount by which the parameter values are to be interpolated between those in config_lo
        and config_hi
    :type config_name_hi: str
    :param input_spiketimes_ms: [default=None] list of list(s) of list(s) of spiketimes to be used as input onto model
        neurons using SpikeGeneratorGroups. See lif.run_simulation() for details
    :type input_spiketimes_ms: [[[float]]] or None
    :param b_verbose: [default=False] if True, additional information is output during the run
    :type b_verbose: bool or int
    :param b_debugging: [default=False] if True, add flag to current line in main log file
    :type b_debugging: bool
    :param b_multiprocessing: if True, run simulations in parallel
    :type b_multiprocessing: bool
    :param batch_size: number of processes (simulations) to run in a batch. None: no batching
    :type batch_size: int or None
    :param recorded_variables: (tuple of) name(s) of the variable(s) to be recorded in StateMonitor and saved to .pkl
    :type recorded_variables: str or tuple
    :param recording_start_ms: [default=0] start recording spikes and states at this timepoint into the simulation
    :type recording_start_ms: float or int
    :param b_save_all: if True, saving_conditions will be ignored, i.e. an ouput .pkl file saved for every sim run.
    :type b_save_all: bool or int
    :param b_query_log_comment: [default=True] if False, don't ask the user to input a comment string for log file
    :type b_query_log_comment: bool
    :param b_run_id_in_filename: [default=True] if False, don't append the run id to directory and file names
        (output files from previous runs of the same config will be overwritten)
    :type b_run_id_in_filename: bool or int
    :return: pathname_out: path to saved output .pkl files
    :rtype: pathname_out: str
    """

    # check inputs
    assert isinstance(config_name_lo, str), "argument 'config_name_lo' must be a string"
    assert isinstance(config_name_hi, str), "argument 'config_name_hi' must be a string"

    # create loggers and get run_id
    log_run_name, run_id = p_io.set_up_loggers('./', config_name_lo, b_debugging=b_debugging,
                                               b_parex=True, b_query_comment=b_query_log_comment)

    # filenames for pickle files for saved results (.pkl)
    if config_name_lo[-5:] == '.json':
        if b_run_id_in_filename:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo[0:-5] + '_' + str(run_id).zfill(4) \
                           + os.path.sep
        else:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo[0:-5] + os.path.sep
    else:
        if b_run_id_in_filename:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo + '_' + str(run_id).zfill(4) \
                           + os.path.sep
        else:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo + os.path.sep

    # check if path already exists and prompt for overwrite
    if os.path.isdir(pathname_out):
        choice = input("! output path " + pathname_out + " already exists. Delete the directory? (y/n)").lower()
        if choice == 'y':
            shutil.rmtree(pathname_out)
        else:
            return

    # load and check config .json files
    config_lo, filename_cfg_lo, _ = p_io.load_config(config_name_lo)
    config_hi, filename_cfg_hi, _ = p_io.load_config(config_name_hi)
    p_io.check_config(config_lo, config_name_lo)

    # copy config files to output directory
    p_io.create_path_to_file(pathname_out)
    shutil.copyfile(filename_cfg_lo, pathname_out + 'config_lo.json')
    shutil.copyfile(filename_cfg_hi, pathname_out + 'config_hi.json')

    # get configurations for all simulation runs
    configs, log_lines, combinations, exp_param_names, all_values_per_param = \
        p_io.get_param_exploration_configs(config_lo, config_hi)

    # output explored parameter values for each simulation to file
    with open(pathname_out + 'param_values.txt', 'w') as file:
        for i_sim in range(len(log_lines)):
            file.write(log_lines[i_sim])

    if b_multiprocessing:
        filename_out_list = []
        for i_sim in range(len(configs)):
            filename_out_list.append(pathname_out + 'parex_' + str(i_sim).zfill(6) + '.pkl')
        lif.run_multiprocess(config_name_lo, configs, filename_out_list, log_name=log_run_name, run_id=run_id,
                             input_spiketimes_ms=input_spiketimes_ms, b_verbose=b_verbose, batch_size=batch_size,
                             recorded_variables=recorded_variables, recording_start_ms=recording_start_ms,
                             b_save_all=b_save_all)
    else:
        # loop through configurations and run simulation
        for i_sim, config_cur in enumerate(configs):
            filename_out = pathname_out + 'parex_' + str(i_sim).zfill(6) + '.pkl'
            lif.run_simulation(config_name_lo, config_preloaded=config_cur, log_name=log_run_name, run_id=run_id,
                               input_spiketimes_ms=input_spiketimes_ms, filename_out_arg=filename_out,
                               recorded_variables=recorded_variables, b_verbose=b_verbose,
                               b_always_save_output=b_save_all, recording_start_ms=recording_start_ms)

    return pathname_out


def sensitivity_analysis(config_name_lo, run_id=None, b_show_figures=True, b_save_figures=False,
                         b_run_id_in_filename=True):
    """Takes parameter exploration config where a parameter (w & n_neurons_per_pop implemented) was varied for two
    synapses/population. Generate 2d matrix of subplots of traces of the neuron of interest in that parameter space,
    that visualizes where certain conditions are met (determined by p_analysis.check_conditions()).
    WARNING: adds axis labels (e.g. 'Excitatory weights [pA]') that are only correct for certain config files.

    :param config_name_lo: filename of original regular .json config file, containing lower limits of all parameters.
        This is used as the name of the /out directory from which results are loaded.
    :type config_name_lo: str
    :type run_id: integer id of the parex simulations run to be analyzed (as used in log files, etc.)
    :type run_id: int or None
    :param b_show_figures: [default=True] if True, figure will not be closed when saving
    :type b_show_figures: bool or int
    :param b_save_figures: [default=False] if True, figure will be saved in fig/
    :type b_save_figures: bool or int
    :param b_run_id_in_filename: [default=True] if False, don't append the run id to directory and file names
        (should be the same setting as in the call to parameter_exploration() that produced the output to be analyzed)
    :type b_run_id_in_filename: bool or int
    :return:
    """

    # check inputs
    assert isinstance(config_name_lo, str), "argument 'config_name_lo' must be a string"
    if b_run_id_in_filename:
        assert isinstance(run_id, int), "if b_run_id_in_filename is True, argument 'run_id' must be an integer"

    # path to pickle files for saved results (.pkl)
    if config_name_lo[-5:] == '.json':
        if b_run_id_in_filename:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo[0:-5] + '_' + str(run_id).zfill(4) \
                           + os.path.sep
        else:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo[0:-5] + os.path.sep
    else:
        if b_run_id_in_filename:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo + '_' + str(run_id).zfill(4) \
                           + os.path.sep
        else:
            pathname_out = BASE_PATH_OUTPUT + 'out' + os.path.sep + config_name_lo + os.path.sep
    assert os.path.isdir(pathname_out), "output path " + pathname_out + " does not exists."

    # get list of .pkl files in output directory
    file_list = p_io.get_file_list(pathname_out, '.pkl')

    # load lo and hi config files and get a list of config dictionaries of all simulations
    config_lo, _, _ = p_io.load_config(pathname_out + 'config_lo.json')
    config_hi, _, _ = p_io.load_config(pathname_out + 'config_hi.json')

    print('loading data and checking contitions met...')
    statemons = []
    spikemons = []
    infos = []
    for i_file, filename in enumerate(file_list):
        states, spikes, spikes_poisson, connectivity, connectivity_generator, config, info = \
            p_io.load_monitors(os.path.join(pathname_out, filename), b_silent=True)
        statemons += states
        spikemons += spikes
        infos += info

    # get values of the parameters varied in the simulations
    _, _, combinations, exp_param_names, all_values_per_param = \
        p_io.get_param_exploration_configs(config_lo, config_hi, b_generate_configs=False)
    x_values = []
    y_values = []
    if 'w' in config_hi['step_size'].keys():
        xlabel = 'Excitatory weights [pA]'
        ylabel = 'Inhibitory weights [pA]'
        unit_factor = 1000
    elif 'n_neurons_per_pop' in config_hi['step_size'].keys():
        xlabel = 'N inhibitory interneurons'
        ylabel = 'N vocal-related input neurons'
        unit_factor = 1
    else:
        xlabel = '?'
        ylabel = '?'
        unit_factor = 1
    for values in all_values_per_param[::-1]:
        if len(values) > 1:
            if not x_values:
                x_values = [round(v * unit_factor) for v in values]
            elif not y_values:
                y_values = [round(v * unit_factor) for v in values]

    # set conditions
    t_baseline = (70, 120)
    t_spikes = (150, 200)
    baseline_cond = (-65, -45)
    n_spikes_cond = (1, 6)

    # check conditions
    conditions_met = p_analysis.check_conditions(spikemons, statemons, infos, config_lo, len(x_values), len(y_values),
                                                 t_baseline=t_baseline, t_spikes=t_spikes,
                                                 baseline_cond=baseline_cond, n_spikes_cond=n_spikes_cond)

    fig_sns, ax_sns = p_plot.plot_sensitivity_traces(statemons, spikemons, infos, config_lo, conditions_met,
                                                     x_values, y_values, xlims=(70, 250), figsize=(13, 7),
                                                     t_baseline=t_baseline, t_spikes=t_spikes,
                                                     baseline_cond=baseline_cond,
                                                     xlabel=xlabel, ylabel=ylabel)
    if b_save_figures:
        if b_run_id_in_filename:
            pathname_fig_sns = BASE_PATH_OUTPUT + 'fig' + os.path.sep + config_name_lo + os.path.sep \
                               + str(run_id).zfill(4) + os.path.sep
        else:
            pathname_fig_sns = BASE_PATH_OUTPUT + 'fig' + os.path.sep + config_name_lo + os.path.sep
        p_io.save_figures(fig_sns, pathname_fig_sns + 'sensitivity_traces',
                          figure_format=FIGURE_FORMATS, b_close_figures=(not b_show_figures))

    return fig_sns, ax_sns
