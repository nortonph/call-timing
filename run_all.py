"""Script that runs all simulations and generates all figures used in the manuscript. [call-time-model]
"""

import os
import gc  # for manual garbage collection
from timeit import default_timer as timer  # used to time overall execution duration

import routine
from mod import lif
from fcn import p_io, p_util
from pub import fig_ms

# this will prevent the following code being executed when this module is imported
if __name__ == "__main__":
    # flags for selectively running parts of the script
    b_run_simulations = True
    b_generate_figures = True
    # run some simulations in parallel using multiple CPU cores (requires more RAM, ~32GB incl. swap)
    b_multiprocessing = False

    # list of filenames (w/o ending .json) of the configuration files (in cfg/) of models to be simulated
    config_names = [
                    'no-inh_free-w',  # No inhibition
                    'tonic-inh_free-w',  # Tonic inhibition
                    'ff-inh_free-w',  # FF inhibition
                    'ff-inh_free-w_exc',  # FF inhibition (Hyperpolarized)
                    'no-inh_free-w_no-spiking',  # No inhibition (no spiking)
                    'tonic-inh_free-w_no-spiking',  # Tonic inhibition (no spiking)
                    'ff-inh_free-w_no-spiking',  # FF inhibition (no spiking)
                    'ff-inh_jam_only_free-w',  # FF inhibition (only auditory-related input)
                    'ff-inh_jam_free-t',  # FF inhibition (auditory and motor input, varying time)
                    'ff-inh_current_plot',  # current plot for supplementary figure
                    'properties-synapses',  # recording of PSPs and PSCs after single spike
                   ]

    config_names_agg_lo = [
                           'ff-inh_jam_only_agg_lo',
                           'no-inh_agg_lo',
                           'tonic-inh_agg_lo',
                           'tonic-inh_agg_low-inh_lo',
                           'ff-inh_agg_lo',
                           'ff-inh_agg_low-inh_lo',
                          ]
    config_names_agg_hi = [
                           'ff-inh_jam_only_agg_hi',
                           'no-inh_agg_hi',
                           'tonic-inh_agg_hi',
                           'tonic-inh_agg_low-inh_hi',
                           'ff-inh_agg_hi',
                           'ff-inh_agg_low-inh_hi',
                           ]

    config_names_lo = [
                      'param_exp_sensitivity_weights_lo',
                      'param_exp_sensitivity_pop-size_lo',
                      ]
    config_names_hi = [
                       'param_exp_sensitivity_weights_hi',
                       'param_exp_sensitivity_pop-size_hi',
                      ]

    start = timer()
    if b_run_simulations:
        # run normal simulations
        for config_name in config_names:
            # create loggers and get name of logger for outputting information during run (creates log files in log/)
            log_run_name, run_id = p_io.set_up_loggers('./', config_name, b_query_comment=False)
            if 'current_plot' in config_name:
                recorded_variables = ('v', 'Ie')
                input_spiketimes = None
            elif 'synapses' in config_name:
                recorded_variables = ('v', 'Ise', 'Isi', 'se', 'si')
                input_spiketimes = [[[0]]]
            else:
                recorded_variables = ('v',)
                input_spiketimes = None
            lif.run_simulation(config_name, log_name=log_run_name, run_id=run_id, recorded_variables=recorded_variables,
                               input_spiketimes_ms=input_spiketimes)
            # clear logger for next simulation
            p_io.clear_loggers()

        # run simulations for aggregates (average trace over 100 runs with different randomized current inputs)
        for idx in range(len(config_names_agg_lo)):
            config_lo, _, _ = p_io.load_config(config_names_agg_lo[idx])
            path_out = routine.parameter_exploration(config_names_agg_lo[idx], config_names_agg_hi[idx], None,
                                                     b_save_all=True, b_multiprocessing=b_multiprocessing,
                                                     b_query_log_comment=False, b_run_id_in_filename=False)
            # save the aggregated monitors containing only data from the neuron of interest to save space
            idx_nrn_oi_abs = p_util.get_abs_from_rel_nrn_idx(config_lo['plot']['idx_nrn_oi_relative'],
                                                             config_lo['plot']['idx_pop_oi'],
                                                             config_lo['misc']['n_neurons_per_pop'])
            states, spikes, _, conn, _, config, info = p_io.aggregate_monitors_from_parex(path_out, idx_nrn_oi_abs)
            filename_subset = path_out + 'aggregate' + os.path.sep + config_names_agg_lo[idx] + \
                '_aggregate_subset_nrn' + str(idx_nrn_oi_abs) + '.pkl'
            p_io.save_monitors(states, spikes, None, conn, [], config, info, filename_subset)
            print('o saved data for single neuron subset of aggregated parex as ' + filename_subset)
            # delete raw output files to save disk space
            file_list = p_io.get_file_list(path_out, file_ending='.pkl')
            for filename in file_list:
                try:
                    os.remove(path_out + filename)
                except OSError as e:
                    print("Error deleting raw output file after aggregate: %s - %s." % (e.filename, e.strerror))
            print('o removed raw .pkl files after aggregate creation from ' + path_out)
            # clear logger for next simulation
            p_io.clear_loggers()

        # run parameter exploration simulations (2d parameter space for sensitivity analysis)
        for idx in range(len(config_names_lo)):
            config_lo, _, _ = p_io.load_config(config_names_lo[idx])
            routine.parameter_exploration(config_names_lo[idx], config_names_hi[idx], None, b_save_all=True,
                                          b_multiprocessing=b_multiprocessing, b_query_log_comment=False,
                                          b_run_id_in_filename=False)
            # clear logger for next simulation
            p_io.clear_loggers()

    if b_generate_figures:
        # generate and save all manuscript figures sequentially
        for f in range(1, 13, 1):
            b_plot_fig = [None] * 13
            b_plot_fig[f] = True
            fig_ms.generate_figures(b_save_figures=True, b_plot_fig=b_plot_fig)
            gc.collect()

    end = timer()
    print("DONE. Overall run time: " + str(round(end-start, 1)) + " seconds.")
