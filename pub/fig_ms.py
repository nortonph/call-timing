"""Functions that generate the figures for the manuscript. Main function: generate_figures(). [call-time-model]
"""

# the published figures were plotted using lualatex and the font Helvetica. If installed, set these to true:
B_USE_LATEX = False
B_USE_HELVETICA = False

import matplotlib
if B_USE_LATEX:
    matplotlib.use('pgf')  # noqa: E402 (must come before importing pyplot)
    from matplotlib.backends.backend_pgf import FigureCanvasPgf
    matplotlib.backend_bases.register_backend('png', FigureCanvasPgf)  # noqa: E402 (must come before importing pyplot)
    matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)  # noqa: E402 (must come before importing pyplot)
import matplotlib.pyplot as plt
import os
import math
import scipy.io
import numpy as np
import brian2 as b2
import matplotlib.gridspec
from scipy.stats import norm

import routine
from fcn import p_io, p_util, p_plot, p_analysis

# globals
C_COLORS = [[0.050, 0.350, 0.800],  # 0 blue
            [0.850, 0.450, 0.000],  # 1 orange
            [0.933, 0.400, 0.550],  # 2 pink
            [0.750, 0.150, 0.150],  # 3 red
            [0.850, 0.750, 0.150],  # 4 yellow
            [0.150, 0.500, 0.500],  # 5 teal
            [0.286, 0.000, 0.573],  # 6 purple
            [0.0, 0.0, 0.0]]  # -1 black
C_COLORS_LIGHT = [[0.300, 0.700, 1.000],  # 0 blue
                  [1.000, 0.650, 0.200],  # 1 orange
                  [1.000, 0.700, 0.800],  # 2 pink
                  [1.000, 0.350, 0.350],  # 3 red
                  [0.950, 0.850, 0.250],  # 4 yellow
                  [0.250, 0.700, 0.700],  # 5 teal
                  [0.686, 0.400, 0.973],  # 6 purple
                  [0.0, 0.0, 0.0]]  # -1 black
COL_SEQ = [5, 3, 4, 0, 1, -1, 6, 2]  # sequence of indices to C_COLORS used for populations by order of appearence
FONTSIZE_XXS = 9
FONTSIZE_XS = 9
FONTSIZE_S = 10
FONTSIZE_M = 12
FONTSIZE_L = 14
FONTSIZE_XL = 16


def generate_figures(b_save_figures=True, b_plot_fig=None):
    """Main function that loads data, calls the functions that plot the individual figures, and saves those.

    :param b_save_figures: If False, figures will not be saved.
    :type b_save_figures: bool
    :param b_plot_fig: 12-element list that determines which of the 11 manuscript figures (incl. suppl. figures) to plot
        setting b_plot_fig[n] to True (or 1), means Figure N will be plotted (see beginning of generate_figures() code
        for ordering of figures). Set all other elements to False or None. If b_plot_fig is None, then all figures will
        be plotted at once (requires a lot of RAM and crashes if not sufficient)
    :type b_plot_fig: list or None
    :return:
    """

    if b_plot_fig is None:
        # selectively plot individual figures by setting the corresponding values to 1 and all others to 0:
        # plot figures      1, 2, 3, 4, 5, sens, sens2, curr, psp, psa, depolarized
        b_plot_fig = [None, 1, 1, 1, 1, 1, 1,    1,     1,    1,   1,   1]
    dpi_fig = 600
    figure_formats = ['png', 'pdf']  # ['png', 'pdf', 'eps']

    # settings necessary for having text of different colors within a string using latex
    # needs matplotlib.use('pgf') before importing pyplot (see above)
    if B_USE_LATEX:
        plt.rcParams.update({
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters (needed for \textbf and \textcolor)
            "pgf.texsystem": "lualatex",  # requires lualatex and texlive-luatex
            "mathtext.fontset": "custom",
            "pgf.preamble": [r'\usepackage{color}', r'\usepackage{fontspec}']})
        if B_USE_HELVETICA:
            plt.rcParams.update({
                "mathtext.rm": "Helvetica",
                "mathtext.it": "Helvetica:italic",
                "mathtext.bf": "Helvetica:bold",
                "pgf.preamble": [r'\usepackage{color}', r'\usepackage{fontspec}',
                                 r'\setsansfont{Helvetica}', r'\setmainfont{Helvetica}']})

    # set font sizes
    plt.rc('font', size=FONTSIZE_M)  # controls default text sizes
    plt.rc('axes', titlesize=FONTSIZE_L)  # fontsize of the axes title
    plt.rc('axes', labelsize=FONTSIZE_S)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=FONTSIZE_XS)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=FONTSIZE_XS)  # fontsize of the tick labels
    plt.rc('legend', fontsize=FONTSIZE_XXS)  # legend fontsize
    plt.rc('figure', titlesize=FONTSIZE_M)  # fontsize of the figure title

    # directories, indices and other parameters
    path_to_traces_example = 'rec/traces/example_traces'
    path_to_traces_spiking = 'rec/traces/PM_spiking'
    path_to_traces_gabazine = 'rec/traces/PM_gabazine'
    path_to_traces_playback_aligned = 'rec/traces/playback_aligned'
    path_to_traces_nif = 'rec/traces/NIf'
    path_to_spiketimes_int = 'rec/spiketimes/interneurons_n-gt-three'
    path_to_spiketimes_spiking = 'rec/spiketimes/PM_spiking'
    path_to_spiketimes_gabazine = 'rec/spiketimes/PM_gabazine'
    path_to_spiketimes_nif = 'rec/spiketimes/NIf'
    filename_nexus_mat = 'rec/spiketimes/nexus_spiketimes/spks_rfrmt.mat'  # final
    filename_nexus_responses_mat = 'rec/spiketimes/nexus_spiketimes/nexus_responses.mat'
    filename_nexus_psth_mat = 'rec/spiketimes/nexus_spiketimes/psth_data.mat'  # psth data from margarida's script
    filename_nexus_responsive_idx = 'rec/spiketimes/nexus_spiketimes/idx_responsive_neurons.mat'
    filename_nexus_putativeint_idx = 'rec/spiketimes/nexus_spiketimes/idx_putativeint.mat'
    filename_nexus_putativeint_responsive_idx = 'rec/spiketimes/nexus_spiketimes/idx_putativeint_responsive.mat'

    # load model data
    # NOTE: IN THIS SCRIPT "MODEL 0" (or mod0, m0, ag0) REFERS TO THE NO-INHIBITION MODEL, "MODEL 1" TO THE
    # FEED_FORWARD INHIBITION MODEL AND "MODEL 2" REFERS TO THE TONIC INHIBITION MODEL
    mod1_name_free_w = 'ff-inh_free-w'
    states_fw, spikes_fw, _, _, _, config_fw, info_fw = load_model_data(mod1_name_free_w)
    if b_plot_fig[1]:
        mod1_name_hyper = 'ff-inh_free-w_exc'
        states_hy, spikes_hy, _, _, _, config_hy, info_hy = load_model_data(mod1_name_hyper)
    if b_plot_fig[1] or b_plot_fig[10]:
        mod1_aggregate_path = 'out/ff-inh_agg_lo/aggregate/ff-inh_agg_lo_aggregate_subset_nrn180.pkl'
        states_ag1, spikes_ag1, _, _, _, _, info_ag1 = p_io.load_monitors(mod1_aggregate_path)
    if b_plot_fig[2]:
        mod0_name_no_ffint = 'no-inh_free-w'
        states_m0, spikes_m0, _, _, _, config_m0, info_m0 = load_model_data(mod0_name_no_ffint)
        mod0_name_no_spiking = 'no-inh_free-w_no-spiking'
        states_m0_nospike, _, _, _, _, config_m0_nospike, info_m0_nospike = load_model_data(mod0_name_no_spiking)
        mod1_name_no_spiking = 'ff-inh_free-w_no-spiking'
        states_m1_nospike, _, _, _, _, config_m1_nospike, info_m1_nospike = load_model_data(mod1_name_no_spiking)
        mod2_name_no_spiking = 'tonic-inh_free-w_no-spiking'
        states_m2_nospike, _, _, _, _, config_m2_nospike, info_m2_nospike = load_model_data(mod2_name_no_spiking)
        mod2_name_tonic_int = 'tonic-inh_free-w'
        states_m2, spikes_m2, _, _, _, config_m2, info_m2 = load_model_data(mod2_name_tonic_int)
    if b_plot_fig[10]:
        mod0_aggregate_path = 'out/no-inh_agg_lo/aggregate/no-inh_agg_lo_aggregate_subset_nrn181.pkl'
        states_ag0, spikes_ag0, _, _, _, _, info_ag0 = p_io.load_monitors(mod0_aggregate_path)
        mod2_aggregate_path = 'out/tonic-inh_agg_lo/aggregate/tonic-inh_agg_lo_aggregate_subset_nrn181.pkl'
        states_ag2, spikes_ag2, _, _, _, _, info_ag2 = p_io.load_monitors(mod2_aggregate_path)
        mod1_agg_low_inh_path = 'out/ff-inh_agg_low-inh_lo/aggregate/ff-inh_agg_low-inh_lo_aggregate_subset_nrn180.pkl'
        states_ag1_lowin, spikes_ag1_lowin, _, _, _, _, info_ag1_lowin = p_io.load_monitors(mod1_agg_low_inh_path)
        mod2_agg_low_inh_path = 'out/tonic-inh_agg_low-inh_lo/aggregate/tonic-inh_agg_low-inh_lo_aggregate_subset_nrn181.pkl'
        states_ag2_lowin, spikes_ag2_lowin, _, _, _, _, info_ag2_lowin = p_io.load_monitors(mod2_agg_low_inh_path)
    if b_plot_fig[4] or b_plot_fig[5]:
        mod1_jam_only_fw = 'ff-inh_jam_only_free-w'
        states_jam_only, spikes_jam_only, _, _, _, config_jam_only, info_jam_only = load_model_data(mod1_jam_only_fw)
        mod1_jam_free_t = 'ff-inh_jam_free-t'
        states_jam_ft, spikes_jam_ft, _, _, _, config_jam_ft, info_jam_ft = load_model_data(mod1_jam_free_t)
        suppr_win = get_suppression_window(config_jam_only, spikes_jam_ft, config_jam_ft, info_jam_ft)
    if b_plot_fig[4]:
        mod1_agg_jam_only = 'out/ff-inh_jam_only_agg_lo/aggregate/ff-inh_jam_only_agg_lo_aggregate_subset_nrn180.pkl'
        states_agg_jam_only, _, _, _, _, _, info_agg_jam_only = p_io.load_monitors(mod1_agg_jam_only)
    if b_plot_fig[6]:
        config_name_sens_weights_lo = 'param_exp_sensitivity_weights_lo'
        run_id_sens_weights = None
    if b_plot_fig[7]:
        config_name_sens_pop_lo = 'param_exp_sensitivity_pop-size_lo'
        run_id_sens_pop = None
    if b_plot_fig[8]:
        mod_current = 'ff-inh_current_plot'
        states_curr, _, _, _, _, config_curr, info_curr = load_model_data(mod_current)
    if b_plot_fig[9]:
        mod_psp = 'properties-synapses'
        states_psp, _, _, _, _, config_psp, info_psp = load_model_data(mod_psp)

    # load recording data
    if b_plot_fig[1] or b_plot_fig[2]:
        traces_example, _, spiketimes_int, _ = load_recording_data(path_to_traces_example, path_to_spiketimes_int)
    if b_plot_fig[1] or b_plot_fig[2] or b_plot_fig[5] or b_plot_fig[10]:
        traces_spiking, _, spiketimes_spiking, _ = load_recording_data(path_to_traces_spiking,
                                                                       path_to_spiketimes_spiking)
    if b_plot_fig[2] or b_plot_fig[5] or b_plot_fig[10]:
        traces_gabazine, _, spiketimes_gabazine, _ = \
            load_recording_data(path_to_traces_gabazine, path_to_spiketimes_gabazine)
    if b_plot_fig[4]:
        traces_nif, _, spiketimes_nif, _ = load_recording_data(path_to_traces_nif, path_to_spiketimes_nif)

    # load behavioral data
    if b_plot_fig[5]:
        call_onset_times = scipy.io.loadmat('rec/behavioral_data/all_call_onset_times.mat')

    # load playback-aligned premotor traces
    if b_plot_fig[4] or b_plot_fig[11]:
        traces_playback_aligned, _, _, _ = load_recording_data(path_to_traces_playback_aligned, None)

    # load neuronexus data with original neuron ids
    if b_plot_fig[3] or b_plot_fig[4]:
        spiketimes_nexus, info_bslcup, playback_name, playback_dur, playback_type, unit_id = \
            p_io.load_neuronexus_spiketimes_from_mat(filename_nexus_mat)
        idx_nexus_responsive = scipy.io.loadmat(filename_nexus_responsive_idx)
        idx_nexus_putativeint = scipy.io.loadmat(filename_nexus_putativeint_idx)
        idx_nexus_putativeint_responsive = scipy.io.loadmat(filename_nexus_putativeint_responsive_idx)
        psths_nexus_matlab = scipy.io.loadmat(filename_nexus_psth_mat)
    if b_plot_fig[3]:
        # load times of significant responses in neuronexus psths, for the horizontal bar plot
        nexus_responses = scipy.io.loadmat(filename_nexus_responses_mat)

    # load embedded images
    if b_plot_fig[1]:
        img_name_zf_calling = 'fig/embed/zf_calling.png'
        img_zf_calling = plt.imread(img_name_zf_calling)
    if b_plot_fig[3] or b_plot_fig[4]:
        img_name_zf_listening = 'fig/embed/zf_listening.png'
        img_zf_listening = plt.imread(img_name_zf_listening)
        img_name_speaker = 'fig/embed/speaker.png'
        img_speaker = plt.imread(img_name_speaker)
    if b_plot_fig[1] or b_plot_fig[3] or b_plot_fig[4]:
        img_name_bubble = 'fig/embed/bubble.png'
        img_bubble = plt.imread(img_name_bubble)

    # plot figures
    pathname_fig = p_io.BASE_PATH_OUTPUT + 'fig' + os.path.sep + 'figures' + os.path.sep
    if b_plot_fig[1]:
        fig1, ax1 = figure1(states_fw, spikes_fw, info_fw, states_hy, info_hy,
                            states_ag1, spikes_ag1, info_ag1, traces_example,
                            spiketimes_int, traces_spiking, spiketimes_spiking, img_zf_calling, img_bubble)
        p_plot.format_fig(fig1, ax1, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig1, pathname_fig + 'fig1', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[2]:
        fig2, ax2 = figure2(states_fw, spikes_fw, info_fw, config_fw, states_m0, spikes_m0, info_m0, config_m0,
                            states_m2, spikes_m2, info_m2, config_m2,
                            states_m0_nospike, info_m0_nospike, config_m0_nospike, states_m1_nospike, info_m1_nospike,
                            config_m1_nospike, states_m2_nospike, info_m2_nospike, config_m2_nospike,
                            traces_example, traces_spiking, spiketimes_spiking, traces_gabazine, spiketimes_gabazine)
        p_plot.format_fig(fig2, ax2, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig2, pathname_fig + 'fig2', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[3]:
        fig3, ax3 = figure3(spiketimes_nexus, unit_id, psths_nexus_matlab, idx_nexus_responsive,
                            idx_nexus_putativeint, idx_nexus_putativeint_responsive, nexus_responses)
        p_plot.format_fig(fig3, ax3, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig3, pathname_fig + 'fig3', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[4]:
        fig4, ax4 = figure4(spikes_jam_only, info_jam_only, config_jam_only, states_agg_jam_only, info_agg_jam_only,
                            spiketimes_nexus, traces_playback_aligned, traces_nif,
                            spiketimes_nif, suppr_win, img_zf_listening, img_bubble, img_speaker)
        p_plot.format_fig(fig4, ax4, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig4, pathname_fig + 'fig4', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[5]:
        fig5, ax5 = figure5(states_jam_only, spikes_jam_only, info_jam_only, config_jam_only,
                            states_jam_ft, spikes_jam_ft, config_jam_ft, info_jam_ft,
                            spiketimes_spiking, spiketimes_gabazine, call_onset_times, suppr_win)
        p_plot.format_fig(fig5, ax5, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig5, pathname_fig + 'fig5', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[6]:  # supplementary figure sensitivity analysis (weights)
        fig_sens, ax_sens = fig_s_sensitivity(config_name_sens_weights_lo, run_id_sens_weights, idx_marked_sim=40,
                                              b_mark_weight_sims=True)
        p_plot.format_fig(fig_sens, ax_sens, b_remove_box=False, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_sens, pathname_fig + 'figS4', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[7]:  # supplementary figure sensitivity analysis (population sizes)
        fig_sens2, ax_sens2 = fig_s_sensitivity(config_name_sens_pop_lo, run_id_sens_pop, idx_marked_sim=71)
        p_plot.format_fig(fig_sens2, ax_sens2, b_remove_box=False, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_sens2, pathname_fig + 'figS1', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[8]:  # supplementary figure currents
        fig_curr, ax_curr = fig_s_current(states_curr, info_curr, config_curr)
        p_plot.format_fig(fig_curr, ax_curr, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_curr, pathname_fig + 'figS2', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[9]:  # supplementary figure post-synaptic currents and potentials
        fig_psp, ax_psp = fig_s_psp(states_psp, config_psp, info_psp)
        p_plot.format_fig(fig_psp, ax_psp, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_psp, pathname_fig + 'figS6', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[10]:  # supplementary figure pre-spike averages of observed and 3 models + dot plot Vm diff
        fig_psa, ax_psa = fig_s_prespike_ramp(states_ag0, spikes_ag0, info_ag0,
                                              states_ag1, spikes_ag1, info_ag1, states_ag2, spikes_ag2, info_ag2,
                                              states_ag1_lowin, spikes_ag1_lowin, info_ag1_lowin,
                                              states_ag2_lowin, spikes_ag2_lowin, info_ag2_lowin,
                                              traces_spiking, spiketimes_spiking, traces_gabazine, spiketimes_gabazine)
        p_plot.format_fig(fig_psa, ax_psa, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_psa, pathname_fig + 'figS3', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)
    if b_plot_fig[11]:  # supplementary figure pre-spike averages of observed and 3 models + dot plot Vm diff
        fig_dep, ax_dep = fig_s_depolarized(traces_playback_aligned)
        p_plot.format_fig(fig_dep, ax_dep, b_remove_box=True, b_remove_legend=False,
                          b_make_ticklabel_font_sansserif=True)
        if b_save_figures:
            p_io.save_figures(fig_dep, pathname_fig + 'figS5', figure_format=figure_formats, dpi=dpi_fig,
                              b_close_figures=True, run_id=info_fw[0]['run_id'], b_watermark=False)


def figure1(states_fw, spikes_fw, info_fw, states_hy, info_hy, states_ag, spikes_ag, info_ag, traces_rec_example,
            spiketimes_rec_int, traces_rec_spiking, spiketimes_rec_spiking, img_zf_calling, img_bubble):
    # figure parameters
    n_rows = 8
    n_cols = 18
    i_rows_trc = [2, 4, 6, 0, 2, 4, 6]
    i_cols_trc = [0, 0, 0, 5, 5, 5, 5]
    n_cols_trc = 4
    i_rows_crc = [0, 2, 4, 6]
    i_cols_crc = [9, 9, 9, 9]
    n_cols_crc = 2
    i_col_psth = 12
    n_cols_psth = 4
    n_rows_default = 2
    x_lim_traces = [-100, 125]
    x_lim_psths = [-100, 125]
    y_lim_traces = [(-68, -12), (-72, 10), (-77, -55), (-55, 0), (-74, 5), (-65, 15), (-85, -60)]
    linewidth = 2
    color_map_rec = p_plot.get_color_list(7, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])
    color_map_mod = p_plot.get_color_list(7, color_map_in=[C_COLORS[c] for c in COL_SEQ])
    color_map_trc = [color_map_rec[1]] + [color_map_rec[3]] + [color_map_rec[2]] + \
        color_map_mod[0:2] + [color_map_mod[3]] + [color_map_mod[2]]

    # file, population & trace indices
    playback_onset_mod = 200
    samp_freq_rec = 40
    id_rec_files = [1, 3, 5]  # 1-based indexing
    idx_rec_trc_oi = [1, 5, 3]
    rec_offset_ms_trc = [200, 100, 200]
    rec_offset_ms_int = [200] * len(spiketimes_rec_int)
    idx_mon_oi_mod = [3, 3, 3, 3]
    idx_pop_oi_mod = [0, 1, 2, 2]
    idx_nrn_oi_mod_rel = [6, 18, 0, 0]
    letter_pop_mod = ['V', '$\mathbf{I_v}$', 'P']
    dur_prespike = 50
    n_rec_traces = len(id_rec_files)
    artificial_spike_heights = [None] * n_rec_traces + [25, 35, 40, None]

    # get absolute from relative model neuron indices
    idx_nrn_oi_mod_abs = []
    for i in range(len(idx_nrn_oi_mod_rel)):
        idx_nrn_oi_mod_abs.append(p_util.get_abs_from_rel_nrn_idx(idx_nrn_oi_mod_rel[i], idx_pop_oi_mod[i],
                                                                  info_fw[0]['population_sizes']))

    # collect traces
    traces_fw, samp_freq_fw = p_util.get_traces_from_monitors(states_fw, b_keep_unit=False)
    traces_hy, samp_freq_hy = p_util.get_traces_from_monitors(states_hy, b_keep_unit=False)
    traces = [traces_rec_example[id_rec_files[0] - 1][idx_rec_trc_oi[0]],
              traces_rec_example[id_rec_files[1] - 1][idx_rec_trc_oi[1]],
              traces_rec_example[id_rec_files[2] - 1][idx_rec_trc_oi[2]],
              traces_fw[idx_mon_oi_mod[0]][idx_nrn_oi_mod_abs[0]],
              traces_fw[idx_mon_oi_mod[1]][idx_nrn_oi_mod_abs[1]],
              traces_fw[idx_mon_oi_mod[2]][idx_nrn_oi_mod_abs[2]],
              traces_hy[idx_mon_oi_mod[3]][idx_nrn_oi_mod_abs[3]]]
    spiketimes_trc = [None] * n_rec_traces + \
                     [spikes_fw[idx_mon_oi_mod[0]].t[spikes_fw[idx_mon_oi_mod[0]].i == idx_nrn_oi_mod_abs[0]] / b2.ms,
                      spikes_fw[idx_mon_oi_mod[1]].t[spikes_fw[idx_mon_oi_mod[1]].i == idx_nrn_oi_mod_abs[1]] / b2.ms,
                      spikes_fw[idx_mon_oi_mod[2]].t[spikes_fw[idx_mon_oi_mod[2]].i == idx_nrn_oi_mod_abs[2]] / b2.ms,
                      None]
    samp_freqs = [samp_freq_rec, samp_freq_rec, samp_freq_rec, samp_freq_fw, samp_freq_fw, samp_freq_fw, samp_freq_hy]
    offset_ms = rec_offset_ms_trc + [playback_onset_mod] * 4

    # set up figure
    fig = plt.figure(1, figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_trc = []
    ax_crc = []
    ax_psth = []

    # region call-marker axis ###################################################################################
    y_lim_call = (0, 2)
    ax_call = plt.subplot2grid((n_rows, n_cols), (0, 0), colspan=n_cols_trc, rowspan=n_rows_default)
    ax_call.plot((0, 0), (0, 1), '--k', linewidth=1)
    x = np.arange(0, 100, 0.1)
    sine = np.sin(x * 0.75)
    gauss = norm.pdf(x, 50, 20) * 50 - 0.05
    wave = (sine * gauss / 2.1 + 0.5)
    # embed zebra finch image
    ax_call_bbox = ax_call.get_window_extent()
    ax_call_axes_ratio = ax_call_bbox.width / ax_call_bbox.height
    ax_call_data_ratio = (x_lim_traces[1] - x_lim_traces[0]) / (y_lim_call[1] - y_lim_call[0])
    inset_height = 1.2
    correction_factor = 0.888888  # apparrently axis gets stretched in finale saved image. this corrects for that
    ax_call.imshow(img_zf_calling, aspect='auto', interpolation='none', extent=(x_lim_traces[0], x_lim_traces[0]
                                                                                + correction_factor * inset_height * ax_call_data_ratio / ax_call_axes_ratio,
                                                                                0, inset_height))
    ax_inset_img = ax_call.inset_axes((x_lim_traces[0] + 50, 0.8, 45, 0.9), transform=ax_call.transData)
    ax_inset_img.imshow(img_bubble, aspect='auto', interpolation='none', extent=(-10, 110, -0.8, 1.50))
    ax_inset_img.plot(x, wave, '-k', linewidth=0.5)
    ax_inset_img.axis('off')
    # plot sound wave
    ax_call.plot(x, wave, '-k', linewidth=1)
    ax_call.text(50, 1.5, 'Call production', verticalalignment='center', horizontalalignment='center')
    ax_call.set_xlim(x_lim_traces)
    ax_call.set_ylim(y_lim_call)
    ax_call.axis('off')
    # plot fake x-axis from call onset onwards
    ax_call.plot((0, 100), (y_lim_call[0], y_lim_call[0]), '-k', clip_on=False, linewidth=0.8)
    for xt in [0, 50, 100]:
        ax_call.plot((xt, xt), (y_lim_call[0], y_lim_call[0] - 0.1), '-k', clip_on=False, linewidth=0.8)
    ax_call.set_title('Observed neurons', fontweight='bold')
    ax_call.text(0.5, 0.94, '(data from Benichov \& Vallentin, 2020)', ha='center', fontsize=FONTSIZE_XXS,
                 transform=ax_call.transAxes)
    # endregion

    # region plot traces ########################################################################################
    for i, trc in enumerate(traces):
        # add artificial spikes by increasing the potential value at spiketimes by a predefined amount
        if artificial_spike_heights[i]:
            for spk in spiketimes_trc[i]:
                samp_spike = int(round(spk / (info_fw[0]['dt'] / b2.ms)))
                trc[samp_spike] = trc[samp_spike] + artificial_spike_heights[i]
        # plot trace
        ax_trc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_trc[i], i_cols_trc[i]),
                                       colspan=n_cols_trc, rowspan=n_rows_default))
        dur_sample = 1 / samp_freqs[i]
        dur_trace = dur_sample * trc.size
        x_values = np.linspace(0, dur_trace, trc.size)
        ax_trc[i].plot((0, 0),
                       (y_lim_traces[i][1] - (y_lim_traces[i][1] - y_lim_traces[i][0]) * 0.15, y_lim_traces[i][0]),
                       color=[0, 0, 0], linewidth=1, linestyle='--')
        ax_trc[i].plot(x_values - offset_ms[i], trc, color=color_map_trc[i], linewidth=linewidth)
        # format axis
        ax_trc[i].set_xlim(x_lim_traces)
        ax_trc[i].set_ylabel('$\mathrm{V_m}$ [mV]')
        if y_lim_traces[i]:
            ax_trc[i].set_ylim(y_lim_traces[i])
        if i_rows_trc[i] < n_rows - n_rows_default:
            ax_trc[i].set_xticklabels([])
        else:
            ax_trc[i].set_xlabel('Time from call onset [ms]')
        if i_rows_trc[i] == 0:
            ax_trc[i].set_title('Model neurons', fontweight='bold')
    # endregion

    # region plot neuron descriptions ###########################################################################
    text_nrn = ['Inhibitory Interneuron', 'Premotor Neuron', 'Premotor Neuron (silent)',
                'Predicted Vocal-related Input Neuron', 'Inhibitory Interneuron', 'Premotor Neuron',
                'Premotor Neuron (silent)']
    for i, txt in enumerate(text_nrn):
        ax_trc[i].text((x_lim_traces[1] + x_lim_traces[0]) / 2, y_lim_traces[i][1], text_nrn[i], va='top', ha='center',
                       fontsize=FONTSIZE_S, fontweight='bold', color=color_map_trc[i])
    # endregion

    # region plot circuit diagrams ##############################################################################
    # new diagrams (full diagrams plotted for each; only relevant elements highlighted):
    x_circuit = [[2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2], [2, 1, 1, 2]]
    y_circuit = [[1.5, 1.5, 0.5, 0.5], [1.5, 1.5, 0.5, 0.5], [1.5, 1.5, 0.5, 0.5], [1.5, 1.5, 0.5, 0.5]]
    conn_circ = [[(0, 1), (1, 2), (1, 3), (2, 3)], [(0, 1), (1, 2), (1, 3), (2, 3)],
                 [(0, 1), (1, 2), (1, 3), (2, 3)], [(0, 1), (1, 2), (1, 3), (2, 3)]]
    conn_type = [[1.5, 1.5, 1.5, -1.5], [1.5, 1.5, 1.5, -1.5], [1.5, 1.5, 1.5, -1.5], [1.5, 1.5, 0.5, -1.5]]
    pop_gray = [.8, .8, .8]
    color_circ = [[[0, 0, 0], color_map_mod[0], pop_gray, pop_gray],
                  [pop_gray, pop_gray, color_map_mod[1], pop_gray],
                  [pop_gray, pop_gray, pop_gray, color_map_mod[3]],
                  [pop_gray, pop_gray, pop_gray, color_map_mod[2]]]
    letters = [[None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2]],
               [None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2]],
               [None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2]],
               [None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2]]]
    color_conn = [[[0, 0, 0], pop_gray, pop_gray, pop_gray],
                  [pop_gray, [0, 0, 0], pop_gray, pop_gray],
                  [pop_gray, pop_gray, [0, 0, 0], [0, 0, 0]],
                  [pop_gray, pop_gray, [0, 0, 0], [0, 0, 0]]]
    ramp_current = np.concatenate((np.zeros(5), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 70 + 1)],
                                   np.linspace(0.1, 0, 10), np.zeros(20)))
    currents = [[ramp_current, None, None, None], [ramp_current, None, None, None],
                [ramp_current, None, None, None], [ramp_current, None, None, None]]
    rectangles = [[1, 0.5, 1, 0], [1, 0.5, 1, 0], [1, 0.5, 1, 0], [1, 0.5, 1, 0]]
    x_lim_crc = (min([min(v) for v in x_circuit]) - 0.5, max([max(v) for v in x_circuit]) + 0.5)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.5, max([max(v) for v in y_circuit]) + 0.5)
    for i in range(len(x_circuit)):
        ax_crc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc[i], i_cols_crc[i]),
                                       colspan=n_cols_crc, rowspan=n_rows_default))
        circuit_diagram(ax_crc[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc,
                        letter_size=FONTSIZE_S, region_rectangle_xywh=rectangles[i], region_label='HVC',
                        connection_colors=color_conn[i])
    # endregion

    # region plot psth of predicted sensorimotor population #####################################################
    ax_psth.append(plt.subplot2grid((n_rows, n_cols), (0, i_col_psth), colspan=n_cols_psth, rowspan=n_rows_default))
    spiketimes_aud = p_util.get_spiketimes_from_monitor(spikes_fw[idx_mon_oi_mod[0]], info_fw[idx_mon_oi_mod[0]], 0)
    t_values_aud, _, psth_aud_smooth = p_analysis.get_psth(spiketimes_aud, info_fw[idx_mon_oi_mod[0]]['sim_time'])
    p_plot.plot_psths([v - playback_onset_mod for v in t_values_aud], psths_smooth=psth_aud_smooth,
                      t_const_event_marker=0, h_ax=ax_psth[0], colors=[color_map_mod[0]], labels=['model'])
    ax_psth[0].set_xticklabels([])
    ax_psth[0].set_xlabel('')
    # endregion

    # region plot psth of interneurons with comparison to recorded interneurons #################################
    ax_psth.append(plt.subplot2grid((n_rows, n_cols), (2, i_col_psth), colspan=n_cols_psth, rowspan=n_rows_default))
    spiketimes_int = p_util.get_spiketimes_from_monitor(spikes_fw[idx_mon_oi_mod[1]], info_fw[idx_mon_oi_mod[1]], 1)
    t_int_mod, _, psth_int_mod_smooth = p_analysis.get_psth(spiketimes_int, info_fw[idx_mon_oi_mod[1]]['sim_time'])
    t_int_rec_offset = []
    psth_int_rec_smooth = []
    for i, spiketimes_rec_int_cur in enumerate(spiketimes_rec_int):
        t_tmp, _, psth_tmp_smooth = p_analysis.get_psth(spiketimes_rec_int_cur, info_fw[idx_mon_oi_mod[1]]['sim_time'])
        t_int_rec_offset.append([v - rec_offset_ms_int[i] for v in t_tmp])
        psth_int_rec_smooth.append(psth_tmp_smooth)
    t_int_all = t_int_rec_offset + [[v - playback_onset_mod for v in t_int_mod]]
    psth_int_smooth_all = psth_int_rec_smooth + [psth_int_mod_smooth]
    n_int = len(spiketimes_rec_int)
    color_rec = [[color_map_rec[1][0], color_map_rec[1][1] + o, color_map_rec[1][2] + o]
                 for o in np.arange(-0.35, 0.3501, 0.7 / (n_int - 1))]
    p_plot.plot_psths(t_int_all, psths_smooth=psth_int_smooth_all, t_const_event_marker=0, h_ax=ax_psth[1],
                      labels=[None] * np.floor((n_int - 1) / 2).astype(int) + ['observed'] +
                             [None] * np.ceil((n_int - 1) / 2).astype(int) + ['model'],
                      colors=color_rec + [color_map_mod[1]], linewidth=[1] * len(t_int_rec_offset) + [2])
    ax_psth[1].set_xlabel('Time from call onset [ms]')
    for i in range(len(ax_psth)):
        ax_psth[i].set_title('')
        ax_psth[i].set_ylabel('Spike rate [Hz]')
        ax_psth[i].set_xlim(x_lim_psths)
    # endregion

    # region plot prespike average of recorded premotor neurons and model neuron (100 run random current average)
    ax_psa = plt.subplot2grid((n_rows, n_cols), (5, i_col_psth), colspan=n_cols_psth, rowspan=3)
    # get and plot recorded traces preceding and aligned to first spike
    smoothed_prespike_averages = []
    n_traces_in_prespike_avg = []
    for i in range(len(traces_rec_spiking)):
        smoothed_timeseries_average, smoothed_timeseries_all, n_traces_included = \
            p_analysis.prespike_trace_average(traces_rec_spiking[i], spiketimes_rec_spiking[i], duration=dur_prespike)
        smoothed_prespike_averages.append(smoothed_timeseries_average)
        n_traces_in_prespike_avg.append(n_traces_included)
        dur_sample = 1 / samp_freq_rec
        dur_trace = dur_sample * smoothed_timeseries_average.size
        x_values = np.linspace(0, dur_trace, smoothed_timeseries_average.size)
        h_rec, = ax_psa.plot(x_values - dur_prespike, smoothed_timeseries_average, color=color_map_rec[3], linewidth=1)
    # get and plot prespike average of aggregate model data
    spiketimes_ms_ag = [p_util.get_spiketimes_from_monitor(spikes_ag[m], info_ag[m], 2) for m in range(len(spikes_ag))]
    traces_ag, samp_freq_ag = p_util.get_traces_from_monitors(states_ag)
    # get prespike trace average
    smoothed_timeseries_ag_avg, _, n_traces_included = p_analysis.prespike_trace_average(
        [traces_ag[i][0] / b2.mV for i in range(len(traces_ag))],
        [spiketimes_ms_ag[i][0] for i in range(len(spiketimes_ms_ag))], duration=dur_prespike,
        b_spiketime_is_onset=True, sampling_frequency_khz=samp_freq_ag, b_smooth=False)
    smoothed_prespike_averages.append(smoothed_timeseries_ag_avg)
    n_traces_in_prespike_avg.append(n_traces_included)
    # plot aggregate model prespike average
    dur_sample = 1 / samp_freq_ag
    dur_trace = dur_sample * smoothed_timeseries_ag_avg.size
    x_values = np.linspace(0, dur_trace, smoothed_timeseries_ag_avg.size)
    h_mod, = ax_psa.plot(x_values - dur_prespike, smoothed_timeseries_ag_avg, color=color_map_mod[3], linewidth=2)
    y_lim_psa = (min([min(v) for v in smoothed_prespike_averages]) - 8, 0)
    ax_psa.legend(handles=(h_rec, h_mod), labels=('observed', 'model'))
    ax_psa.set_xlim((-dur_prespike, 0))
    ax_psa.set_ylim(y_lim_psa)
    ax_psa.set_xlabel('Time from first spike [ms]')
    ax_psa.set_ylabel('$\mathrm{V_m}$ [mV]\nrelative to spike onset')
    # endregion

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    # add sub-figure labels
    ax_call.annotate('A', xy=(-0.23, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_trc[3].annotate('B', xy=(-0.23, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc[0].annotate('C', xy=(-0.23, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psth[0].annotate('D', xy=(-0.23, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psa.annotate('E', xy=(-0.23, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    return fig, [ax_call] + ax_trc + ax_crc + ax_psth + [ax_psa]


def figure2(states_fw, spikes_fw, info_fw, config_fw, states_m0, spikes_m0, info_m0, config_m0,
            states_m2, spikes_m2, info_m2, config_m2,
            states_m0_nospike, info_m0_nospike, config_m0_nospike, states_m1_nospike, info_m1_nospike,
            config_m1_nospike, states_m2_nospike, info_m2_nospike, config_m2_nospike,
            traces_rec_example, traces_spiking, spiketimes_spiking, traces_gabazine, spiketimes_gabazine):
    # figure parameters
    n_rows = 23
    n_cols = 19
    i_rows_trc = [0, 2, 5, 7, 10, 12, 15, 15]
    i_cols_trc = [4, 4, 4, 4, 4, 4, 1, 6]
    n_cols_trc = 4
    i_rows_crc = [0, 2, 5, 7, 10, 12]
    i_cols_crc = [8, 8, 8, 8, 8, 8]
    n_cols_crc = 2
    i_rows_crc_big = [0, 5, 10]
    i_cols_crc_big = [0, 0, 0]
    n_rows_crc_big = 4
    n_cols_crc_big = 3
    i_rows_tpr = [0, 5, 10]
    i_cols_tpr = [11, 11, 11]
    n_rows_tpr = 2
    n_cols_tpr = 3
    i_rows_lat = [2, 7, 12]
    i_cols_lat = [11, 11, 11]
    n_rows_lat = 2
    n_cols_lat = 3
    i_rows_psa = [15]
    i_cols_psa = [11]
    n_rows_psa = 2
    n_cols_psa = 3
    n_rows_default = 2
    x_lim_traces = [-100, 75]
    y_lim_traces = [None, None, None, None, None, None, None, None]
    linewidth = 2

    # file, population & trace indices
    playback_onset_mod = 200
    samp_freq_rec = 40
    id_rec_files = [3, 4]  # 1-based indexing
    idx_rec_trc_oi = [5, 1]
    rec_offset_ms_trc = [100, 100]
    idx_mon_oi_mod = [3, 9, 3, 9, 3, 9]
    idx_pop_oi_mod = [3, 3, 3, 3, 2, 2]
    idx_nrn_oi_mod_rel = [0, 0, 0, 0, 0, 0]
    letter_pop_mod = ['V', '$\mathbf{I_v}$', 'P', 'P', 'P', 'T']
    n_rec_traces = len(id_rec_files)
    artificial_spike_heights = [40, 40, 40, 40, 40, 40] + [None] * n_rec_traces
    dur_prespike = 50
    t1_prespike = 0  # this minus dur_prespike in the plot; same for t2
    t2_prespike = dur_prespike - 10
    colormap_traces = 'plasma'
    cmap = plt.cm.get_cmap(colormap_traces)

    # customize colormap
    cm = np.array([cmap(i) for i in np.arange(0, 1, 0.01)])
    cm[0:27, 1] = cm[range(54, 27, -1), 1]
    cm[0:33, 2] = np.linspace(0.8, 0.622, 33)
    cmext = np.ones((14, 4))
    cmext[0:14, 2] = np.linspace(0.86, 0.8, 14)
    cmext[0:14, 1] = cm[range(68, 54, -1), 1]
    cmext[0:14, 0] = np.linspace(0, cm[0, 0], 14)
    cmnew = np.concatenate((cmext, cm))
    cmap = matplotlib.colors.ListedColormap(cmnew)

    color_map_rec = p_plot.get_color_list(7, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])
    color_map_mod = p_plot.get_color_list(7, color_map_in=[C_COLORS[c] for c in COL_SEQ])
    # note: first two runs left out => remove "-2" to undo
    color_map_trc = [cmap((idx_mon_oi_mod[0] - 2) / (len(states_m0_nospike) - 1 - 2)),  # same as colormap
                     cmap((idx_mon_oi_mod[1] - 2) / (len(states_m0_nospike) - 1 - 2)),
                     cmap((idx_mon_oi_mod[2] - 2) / (len(states_m1_nospike) - 1 - 2)),
                     cmap((idx_mon_oi_mod[3] - 2) / (len(states_m1_nospike) - 1 - 2)),
                     cmap((idx_mon_oi_mod[4] - 2) / (len(states_m2_nospike) - 1 - 2)),
                     cmap((idx_mon_oi_mod[5] - 2) / (len(states_m2_nospike) - 1 - 2))] + color_map_rec[3:5]
    # color_map_trc = color_map_mod[3:5] + color_map_mod[3:5] + color_map_mod[3:5] + color_map_rec[3:5]  # blue/orange

    # get absolute from relative model neuron indices
    idx_nrn_oi_mod_abs = []
    pop_szs = [info_m0[0]['population_sizes']] * 2 + [info_m2[0]['population_sizes']] * 2 + \
              [info_fw[0]['population_sizes']] * 2
    for i in range(len(idx_nrn_oi_mod_rel)):
        idx_nrn_oi_mod_abs.append(p_util.get_abs_from_rel_nrn_idx(idx_nrn_oi_mod_rel[i], idx_pop_oi_mod[i], pop_szs[i]))

    # collect traces
    traces_fw, samp_freq_fw = p_util.get_traces_from_monitors(states_fw, b_keep_unit=False)
    traces_m0, samp_freq_m0 = p_util.get_traces_from_monitors(states_m0, b_keep_unit=False)
    traces_m2, samp_freq_m2 = p_util.get_traces_from_monitors(states_m2, b_keep_unit=False)
    traces = [traces_m0[idx_mon_oi_mod[0]][idx_nrn_oi_mod_abs[0]],
              traces_m0[idx_mon_oi_mod[1]][idx_nrn_oi_mod_abs[1]],
              traces_m2[idx_mon_oi_mod[2]][idx_nrn_oi_mod_abs[2]],
              traces_m2[idx_mon_oi_mod[3]][idx_nrn_oi_mod_abs[3]],
              traces_fw[idx_mon_oi_mod[4]][idx_nrn_oi_mod_abs[4]],
              traces_fw[idx_mon_oi_mod[5]][idx_nrn_oi_mod_abs[5]],
              traces_rec_example[id_rec_files[0] - 1][idx_rec_trc_oi[0]],
              traces_rec_example[id_rec_files[1] - 1][idx_rec_trc_oi[1]]]
    spiketimes_trc = [spikes_m0[idx_mon_oi_mod[0]].t[spikes_m0[idx_mon_oi_mod[0]].i == idx_nrn_oi_mod_abs[0]] / b2.ms,
                      spikes_m0[idx_mon_oi_mod[1]].t[spikes_m0[idx_mon_oi_mod[1]].i == idx_nrn_oi_mod_abs[1]] / b2.ms,
                      spikes_m2[idx_mon_oi_mod[2]].t[spikes_m2[idx_mon_oi_mod[2]].i == idx_nrn_oi_mod_abs[2]] / b2.ms,
                      spikes_m2[idx_mon_oi_mod[3]].t[spikes_m2[idx_mon_oi_mod[3]].i == idx_nrn_oi_mod_abs[3]] / b2.ms,
                      spikes_fw[idx_mon_oi_mod[4]].t[spikes_fw[idx_mon_oi_mod[4]].i == idx_nrn_oi_mod_abs[4]] / b2.ms,
                      spikes_fw[idx_mon_oi_mod[5]].t[spikes_fw[idx_mon_oi_mod[5]].i == idx_nrn_oi_mod_abs[5]] / b2.ms,
                      None, None]
    # note: first two runs left out => remove "-2" to undo
    inh_weights = [
        info_m0[2:][idx_mon_oi_mod[0] - 2]['free_parameter_values'][config_m0['plot']['idx_synpop_oi_for_fp']] * 1000,
        info_m0[2:][idx_mon_oi_mod[1] - 2]['free_parameter_values'][config_m0['plot']['idx_synpop_oi_for_fp']] * 1000,
        info_fw[2:][idx_mon_oi_mod[2] - 2]['free_parameter_values'][config_fw['plot']['idx_synpop_oi_for_fp']] * 1000,
        info_fw[2:][idx_mon_oi_mod[3] - 2]['free_parameter_values'][config_fw['plot']['idx_synpop_oi_for_fp']] * 1000,
        info_m2[2:][idx_mon_oi_mod[4] - 2]['free_parameter_values'][config_m2['plot']['idx_synpop_oi_for_fp']] * 1000,
        info_m2[2:][idx_mon_oi_mod[5] - 2]['free_parameter_values'][config_m2['plot']['idx_synpop_oi_for_fp']] * 1000]
    samp_freqs = [samp_freq_m0, samp_freq_m0, samp_freq_fw, samp_freq_fw, samp_freq_m0, samp_freq_m0,
                  samp_freq_rec, samp_freq_rec]
    offset_ms = [playback_onset_mod] * 6 + rec_offset_ms_trc

    # set up figure
    fig = plt.figure(2, figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_trc = []
    ax_crc = []
    ax_crc_big = []
    ax_tpr = []
    ax_lat = []
    ax_psa = []

    # region plot traces ########################################################################################
    for i, trc in enumerate(traces):
        # add artificial spikes by increasing the potential value at spiketimes by a predefined amount
        if artificial_spike_heights[i]:
            for spk in spiketimes_trc[i]:
                samp_spike = int(round(spk / (info_fw[0]['dt'] / b2.ms)))
                trc[samp_spike] = trc[samp_spike] + artificial_spike_heights[i]
        # plot trace
        ax_trc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_trc[i], i_cols_trc[i]),
                                       colspan=n_cols_trc, rowspan=n_rows_default))
        dur_sample = 1 / samp_freqs[i]
        dur_trace = dur_sample * trc.size
        x_values = np.linspace(0, dur_trace, trc.size)
        ax_trc[i].axvline(0, color=[0, 0, 0], linewidth=1, linestyle='--')
        ax_trc[i].plot(x_values - offset_ms[i], trc, color=color_map_trc[i], linewidth=linewidth)
        # format axis
        ax_trc[i].set_xlim(x_lim_traces)
        ax_trc[i].set_ylabel('$\mathrm{V_m}$ [mV]')
        if y_lim_traces[i]:
            ax_trc[i].set_ylim(y_lim_traces[i])
        # remove xticklabels from every second plot (upper trace in each subfigure)
        if i % 2 == 0 and i != len(traces) - 2:
            ax_trc[i].set_xticklabels([])
        else:
            ax_trc[i].set_xlabel('Time from call onset [ms]')
        # print inhibitory weights
        if i < 6:
            ax_trc[i].text(50, 0, str(int(np.rint(inh_weights[i]))) + ' pA', ha='left',
                           va='top', color=color_map_trc[i], size=FONTSIZE_S)
    ax_trc[-2].text(0.05, 0.8, 'control', size=FONTSIZE_S, color=color_map_rec[3], transform=ax_trc[-2].transAxes)
    ax_trc[-1].text(0.05, 0.8, 'gabazine', size=FONTSIZE_S, color=color_map_rec[4], transform=ax_trc[-1].transAxes)
    ax_trc[-2].text(0.35, 1.13, 'Observed neurons', fontweight='bold', ha='center', va='bottom',
                    fontsize=FONTSIZE_L, transform=ax_trc[-2].transAxes)
    ax_trc[-2].text(1.15, 1.12, '(data from Benichov \& Vallentin, 2020)', ha='center', va='bottom',
                    fontsize=FONTSIZE_XXS, transform=ax_trc[-2].transAxes)
    ax_trc[0].set_title('Model neurons', fontweight='bold')
    # endregion

    # region plot small circuit diagrams next to traces #########################################################
    # alternative diagram with no-inhibition model PMs receiving motor-related input as well
    x_circuit = [[1, 1, 2], [1, 1, 2],
                 [1, 1, 2, 2], [1, 1, 2, 2],
                 [1, 1, 2], [1, 1, 2]]
    y_circuit = [[1.5, 0.5, 0.5], [1.5, 0.5, 0.5],
                 [1.5, 0.5, 1.5, 0.5], [1.5, 0.5, 1.5, 0.5],
                 [1.5, 0.5, 0.5], [1.5, 0.5, 0.5]]
    conn_circ = [[(0, 1), (0, 2)], [(0, 1), (0, 2)],
                 [(0, 1), (0, 3), (2, 3)], [(0, 1), (0, 3), (2, 3)],
                 [(0, 1), (0, 2), (1, 2)], [(0, 1), (0, 2), (1, 2)]]
    conn_type = [[1.5, 0.5], [1.5, 0.5],
                 [1.5, 1.5, -1.5], [1.5, 1.5, -1.0],
                 [1.5, 1.5, -1.5], [1.5, 1.5, -1.0]]
    conn_linestyle = [['-', '-'], ['-', '-'], ['-', '-', '-'], ['-', '-', ':'], ['-', '-', '-'], ['-', '-', ':']]
    color_circ = [[color_map_mod[0], color_map_mod[1], color_map_trc[0]],
                  [color_map_mod[0], color_map_mod[1], color_map_trc[1]],
                  [color_map_mod[0], color_map_mod[1], color_map_mod[5], color_map_trc[2]],
                  [color_map_mod[0], color_map_mod[1], color_map_mod[5], color_map_trc[3]],
                  [color_map_mod[0], color_map_mod[1], color_map_trc[4]],
                  [color_map_mod[0], color_map_mod[1], color_map_trc[5]]]
    letters = [[letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[4]],
               [letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[5], letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[5], letter_pop_mod[4]],
               [letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[4]]]
    currents = [None] * len(x_circuit)
    # set axis limits and plot
    x_lim_crc = (min([min(v) for v in x_circuit]) - 0.5, max([max(v) for v in x_circuit]) + 0.5)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.5, max([max(v) for v in y_circuit]) + 0.5)
    for i in range(len(x_circuit)):
        ax_crc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc[i], i_cols_crc[i]),
                                       colspan=n_cols_crc, rowspan=n_rows_default))
        circuit_diagram(ax_crc[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc,
                        letter_size=FONTSIZE_S, linestyle=conn_linestyle[i])
    # endregion

    # region plot big circuit diagrams on top ###################################################################
    # alternative diagram with no-inhibition model PMs receiving motor-related input as well
    x_circuit = [[1, 1, 1, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2]]
    y_circuit = [[1.5, 0.5, 2.5, .5], [1.5, 0.5, 2.5, 1.5, .5], [1.5, 0.5, 2.5, .5]]
    conn_circ = [[(0, 1), (0, 3), (2, 0)], [(0, 1), (0, 4), (3, 4), (2, 0)], [(0, 1), (0, 3), (1, 3), (2, 0)]]
    conn_type = [[1.5, 0.5, 1.5], [1.5, 1.5, -1.5, 1.5], [1.5, 1.5, -1.5, 1.5]]
    conn_colors = [['k', 'k', 'k'], ['k', 'k', [1, 0, 0.4], 'k'], ['k', 'k', [1, 0, 0.4], 'k']]
    color_circ = [[color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[3]],
                  [color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[5], color_map_mod[3]],
                  [color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[3]]]
    letters = [[letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[5], letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[3]]]
    ramp_current = np.concatenate((np.zeros(5), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 70 + 1)],
                                   np.linspace(0.1, 0, 10), np.zeros(20)))
    currents = [[None, None, ramp_current, None], [None, None, ramp_current, None, None],
                [None, None, ramp_current, None]]
    # set axis limits and plot
    x_lim_crc = (min([min(v) for v in x_circuit]) - 0.7, max([max(v) for v in x_circuit]) + 0.3)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.3, max([max(v) for v in y_circuit]) + 0.7)
    text_mod = ['No inhibition', 'Tonic inhibition', 'Feed-forward\ninhibition']
    for i in range(len(x_circuit)):
        ax_crc_big.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc_big[i], i_cols_crc_big[i]),
                                           colspan=n_cols_crc_big, rowspan=n_rows_crc_big))
        circuit_diagram(ax_crc_big[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc,
                        letter_size=FONTSIZE_XL, connection_colors=conn_colors[i])
        ax_crc_big[i].text(1.5, 3.555, text_mod[i], va='top', ha='center', fontsize=FONTSIZE_M, fontweight='bold')
    # endregion

    # region plot traces per run without spike threshold ##############################################
    spike_threshold = -40
    # note: first two runs left out
    statemons = [states_m0_nospike[2:], states_m2_nospike[2:], states_m1_nospike[2:]]
    configs = [config_m0_nospike, config_m2_nospike, config_m1_nospike]
    infos = [info_m0_nospike[2:], info_m2_nospike[2:], info_m1_nospike[2:]]
    pops_oi = [3, 3, 2]
    xlims_tpr_lat = (-55, 0.5)
    for i in range(len(statemons)):
        ax_tpr.append(plt.subplot2grid((n_rows, n_cols), (i_rows_tpr[i], i_cols_tpr[i]),
                                       colspan=n_cols_tpr, rowspan=n_rows_tpr))
        ax_tpr[i].set_clip_on(False)
        p_plot.plot_traces_per_run(statemons[i], configs[i], infos[i], b_spike_threshold=False, linewidth_traces=1,
                                   neuron_idx=0, population_idx=pops_oi[i], t_offset_ms=offset_ms[i], h_ax=ax_tpr[i],
                                   colormap=cmap)
        ax_tpr[i].plot(xlims_tpr_lat, [spike_threshold, spike_threshold], linestyle='-', color=[0, 0, 0], linewidth=1)
        ax_tpr[i].get_legend().remove()
        ax_tpr[i].tick_params(labelright=False)
        ax_tpr[i].set_title('')
        ax_tpr[i].set_xlim(xlims_tpr_lat)
        nrn_idx_abs = p_util.get_abs_from_rel_nrn_idx(0, pops_oi[i], infos[i][0]['population_sizes'])
        i_plot_l = np.where(xlims_tpr_lat[0] < statemons[i][0].t / b2.ms - offset_ms[i])[0][0]
        i_plot_r = np.where(xlims_tpr_lat[1] > statemons[i][0].t / b2.ms - offset_ms[i])[0][-1]
        ax_tpr[i].set_ylim(np.min(statemons[i][0].v[nrn_idx_abs, i_plot_l:i_plot_r] / b2.mV) - 1.5,
                           np.max(statemons[i][-1].v[nrn_idx_abs, i_plot_l:i_plot_r] / b2.mV) + 1.5)
        ax_tpr[i].set_ylabel('$\mathrm{V_m}$ [mV]', weight='regular')
        ax_tpr[i].get_xaxis().set_visible(False)
        ax_tpr[i].spines['bottom'].set_visible(False)
    # endregion

    # region plot latency of first spike versus inhibitory weights ##############################################
    # note: first two runs left out
    spikemons = [spikes_m0[2:], spikes_m2[2:], spikes_fw[2:]]
    configs = [config_m0, config_m2, config_fw]
    infos = [info_m0[2:], info_m2[2:], info_fw[2:]]
    pops_oi = [idx_pop_oi_mod[0], idx_pop_oi_mod[2], idx_pop_oi_mod[4]]
    syn_oi = [config_m0['plot']['idx_synpop_oi_for_fp'], config_m2['plot']['idx_synpop_oi_for_fp'],
              config_fw['plot']['idx_synpop_oi_for_fp']]
    for i in range(len(spikemons)):
        nrn_idx_abs = p_util.get_abs_from_rel_nrn_idx(0, pops_oi[i], infos[i][0]['population_sizes'])
        ax_lat.append(plt.subplot2grid((n_rows, n_cols), (i_rows_lat[i], i_cols_lat[i]),
                                       colspan=n_cols_lat, rowspan=n_rows_lat))
        idx_colored = list(range(len(spikemons[i])))  # all dots in viridis
        colors_lat = [cmap(c) for c in np.linspace(0, 1, len(spikemons[i]))]
        p_plot.plot_t_first_spike_1d(spikemons[i], configs[i], infos[i], nrn_idx_abs, syn_oi[i], h_ax=ax_lat[i],
                                     param_unit=b2.pA, t_const_event_marker=configs[i]['misc']['playback_start'],
                                     idx_mons_colored=idx_colored, colors=colors_lat, dot_size=3)
        ax_lat[i].set_title('')
        ax_lat[i].set_xlim(xlims_tpr_lat)
        ax_lat[i].set_ylabel('Inhibitory\nweights [pA]', weight='regular')
        ax_lat[i].set_xlabel('Time of first spike [ms]', weight='regular')
        # connecting lines between threshoold crossings and dots on latency plot. * 1000 to go from Amp to pA
        t_first_spike = p_analysis.get_t_first_spike_mod(spikemons[i], infos[i], nrn_idx_abs)
        con = []
        for m, t in enumerate(t_first_spike):
            x_t = t - offset_ms[i]
            con.append(matplotlib.patches.ConnectionPatch(xyA=(x_t, spike_threshold),
                                                          xyB=(
                                                          x_t, infos[i][m]['free_parameter_values'][syn_oi[i]] * 1000),
                                                          coordsA="data", coordsB="data", axesA=ax_tpr[i],
                                                          axesB=ax_lat[i],
                                                          color=cmap(m / (len(t_first_spike) - 1))))
        # plot connections in reversed order (i.e. same plotting order as traces above)
        for c in list(reversed(con)):
            ax_lat[i].add_artist(c)
        # plot arrows marking the example traces and the corresponding inhibitory weights
        # note: first two runs left out => remove "-2" to undo
        for a in range(2):
            inh_weight = infos[i][idx_mon_oi_mod[i * 2 + a] - 2]['free_parameter_values'][syn_oi[i]] * 1000
            arr_x_start = t_first_spike[idx_mon_oi_mod[i * 2 + a] - 2] - offset_ms[i] + 4.5
            arr_y_start = inh_weight + 3.5
            arr_color = cmap((idx_mon_oi_mod[i * 2 + a] - 2) / (len(t_first_spike) - 1))
            ax_lat[i].arrow(arr_x_start, arr_y_start, -3, -2.4, width=0.1, head_length=1.2, head_width=1.4,
                            length_includes_head=True, fc=arr_color, ec=arr_color)
            ax_lat[i].text(arr_x_start - 0.5, arr_y_start + 2.5, str(int(np.rint(inh_weight))) + ' pA', ha='left',
                           va='center',
                           color=arr_color, size=FONTSIZE_XS)
    # endregion

    # region plot prespike averages of recorded premotor neurons and model neurons (100 run random current avg.)
    traces_rec_all = [traces_spiking, traces_gabazine]
    spiketimes_rec_all = [spiketimes_spiking, spiketimes_gabazine]
    i_traces_rec_psa = [0, 1, 0, 1, 0, 1]
    i_spiketimes_rec_psa = [0, 1, 0, 1, 0, 1]
    i_colors_psa = [3, 4, 3, 4, 3, 4]
    for p in range(len(i_cols_psa)):
        ax_psa.append(plt.subplot2grid((n_rows, n_cols), (i_rows_psa[p], i_cols_psa[p]),
                                       colspan=n_cols_psa, rowspan=n_rows_psa))
        y_lim_psa = (-30, 0)
        ax_psa[p].set_xlim((-dur_prespike, 0))
        ax_psa[p].set_ylim(y_lim_psa)
        ax_psa[p].set_xlabel('Time from first spike [ms]')
        ax_psa[p].set_ylabel('$\mathrm{V_m}$ [mV]\nrel. to spike onset')
        # for every second plot remove y axis labels
        if len(i_cols_psa) > 1 and p % 2 == 0:
            ax_psa[p].set_xlabel('')
            ax_psa[p].set_xticklabels('')

    # get and plot recorded traces preceding and aligned to first spike
    h_rec_all = []
    h_avg_all = []
    averages = []
    for c in range(len(i_traces_rec_psa)):
        i_ax = 0  # set to c to plot in 2 separate axes
        smoothed_prespike_averages = []
        n_traces_in_prespike_avg = []
        for i in range(len(traces_rec_all[i_traces_rec_psa[c]])):
            smoothed_timeseries_average, smoothed_timeseries_all, n_traces_included = \
                p_analysis.prespike_trace_average(traces_rec_all[i_traces_rec_psa[c]][i],
                                                  spiketimes_rec_all[i_spiketimes_rec_psa[c]][i], duration=dur_prespike)
            smoothed_prespike_averages.append(smoothed_timeseries_average)
            n_traces_in_prespike_avg.append(n_traces_included)
            dur_sample = 1 / samp_freq_rec
            dur_trace = dur_sample * smoothed_timeseries_average.size
            x_values = np.linspace(0, dur_trace, smoothed_timeseries_average.size)
            h_rec, = ax_psa[i_ax].plot(x_values - dur_prespike, smoothed_timeseries_average,
                                       color=color_map_rec[i_colors_psa[c]] + [0.8], linewidth=0.5)
        averages.append(np.mean(smoothed_prespike_averages, 0))
        h_rec_all.append(h_rec)
    # plot averages
    for c in range(len(i_traces_rec_psa)):
        h_avg, = ax_psa[i_ax].plot(x_values - dur_prespike, averages[c],
                                   color=color_map_mod[i_colors_psa[c]], linewidth=2)
        h_avg_all.append(h_avg)
    for p in range(len(i_cols_psa)):
        ax_psa[p].legend(handles=[(h_rec_all[0], h_rec_all[1]), (h_avg_all[0], h_avg_all[1])],
                         labels=['observed', 'average'], loc='upper center', bbox_to_anchor=(0, 0.15, 1, 1.15),
                         handlelength=1, ncol=2, handler_map={tuple: matplotlib.legend_handler.HandlerTuple(None)})
    # endregion

    # add sub-figure labels
    ax_crc_big[0].annotate('A', xy=(-0.1, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc_big[1].annotate('C', xy=(-0.1, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc_big[2].annotate('E', xy=(-0.1, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_tpr[0].annotate('B', xy=(-0.4, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_tpr[1].annotate('D', xy=(-0.4, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_tpr[2].annotate('F', xy=(-0.4, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_trc[6].annotate('G', xy=(-0.22, 1.13), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psa[0].annotate('H', xy=(-0.4, 1.13), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    return fig, ax_trc + ax_crc + ax_crc_big + ax_tpr + ax_lat + ax_psa


def figure3(spiketimes_nexus, unit_id, psths_nexus_matlab, idx_nexus_responsive, idx_nexus_putativeint,
            idx_nexus_putativeint_responsive, nexus_responses):
    # figure parameters
    n_rows = 30
    n_cols = 20
    i_rows_psth = [0, 2, 4]  # psths of Neuronexus recordings
    i_cols_psth = [0, 0, 0]
    n_cols_psth = 3
    i_rows_mtrx = [7, 7, 13, 13]  # matrix plot of all nexus psths
    i_cols_mtrx = [0, 4, 0, 4]
    n_rows_mtrx = [5, 5, 4, 4]
    n_cols_mtrx = [3, 3, 3, 3]
    i_rows_line = [18, 20]  # matrix plot of all nexus psths
    i_cols_line = [7, 7]
    n_rows_line = [2, 2]
    n_cols_line = [1, 1]
    i_rows_bar = [18, 18]  # horizontal bar plot
    i_cols_bar = [0, 4]
    n_rows_bar = 3
    n_cols_bar = 3
    i_rows_sum = [21, 21]  # sum of horizontal bar plot
    i_cols_sum = [0, 4]
    n_rows_sum = 1
    n_cols_sum = 3
    i_rows_hist = [0]  # histogram of baseline activities
    i_cols_hist = [4]
    n_rows_hist = 2
    n_cols_hist = 3
    n_rows_default = 2
    t_baseline_psth_norm = (-500, 0)  # for normalizing psths to baseline. Set to None to not normalize
    x_lim_psths = [-100, 500]
    y_lim_psths = [0, 3]

    # file, population & trace indices
    id_rec_nrn_nxs = [2, 4, 123]  # final (NOTE: this is 1-based index, variables starting with idx_ are 0-based!)
    idx_responsive = idx_nexus_responsive['idx_responsive_neurons_out_of_min20trials'][0] - 1
    idx_putativeint = idx_nexus_putativeint['idx_putative_interneurons_out_of_min20trials'][0] - 1
    idx_putativeint_responsive = \
        idx_nexus_putativeint_responsive['idx_putative_interneurons_responsive_out_of_min20trials'][0] - 1
    pathname_out = p_io.BASE_PATH_OUTPUT + 'fig' + os.path.sep + 'fig_ms' + os.path.sep + 'spiketimes' + os.path.sep + \
        'nexus_psth' + os.path.sep

    # set up figure
    fig = plt.figure(3, figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_psth = []
    ax_psth_spks = []
    ax_mtrx = []
    ax_cb = []
    ax_line = []
    ax_bar = []
    ax_sum = []
    ax_hist = []

    # region get data of all nexus psths ############################################################################
    b_sort_psths = True
    # normalize psths as loaded from margarida's matlab script
    psth_loaded = np.squeeze(np.stack(psths_nexus_matlab['psth'][0]))
    psth_t = np.squeeze(np.stack(psths_nexus_matlab['time_n'][0])) * 1000
    psth_sem = np.squeeze(np.stack(psths_nexus_matlab['psth_error'][0]))
    # shift psths so baseline is at 0, and normalize to absolute maximum (max or min equals 1/-1, whichever larger)
    psth_t_truncated = []
    psth_truncated = []
    psth_sem_truncated = []
    psths_nexus_blzero_norm = []
    psths_nexus_blnorm = []
    psth_sem_blnorm = []
    psth_baseline_std_norm = []
    psth_baseline_mean = []
    for i, p in enumerate(psth_loaded):
        samp_baseline_start = np.where(psth_t[i] >= t_baseline_psth_norm[0])[0][0]
        samp_baseline_end = np.where(psth_t[i] <= t_baseline_psth_norm[1])[0][-1]
        baseline_activity = np.mean(p[samp_baseline_start:samp_baseline_end])
        baseline_std = np.std(p[samp_baseline_start:samp_baseline_end])
        # truncate psths to x_lim_psths (set above)
        samp_xmin_psths = np.where(psth_t[i] >= x_lim_psths[0])[0][0]
        samp_xmax_psths = np.where(psth_t[i] >= x_lim_psths[1])[0][0]
        psth_truncated.append(p[samp_xmin_psths:samp_xmax_psths])
        psth_t_truncated.append(psth_t[i][samp_xmin_psths:samp_xmax_psths])
        psth_sem_truncated.append(psth_sem[i][samp_xmin_psths:samp_xmax_psths])
        psth_baseline_zero = np.asarray(psth_truncated[i]) - baseline_activity
        psths_nexus_blzero_norm.append(psth_baseline_zero / max(np.abs(psth_baseline_zero)))
        psths_nexus_blnorm.append(np.asarray(psth_truncated[i]) / baseline_activity)
        psth_sem_blnorm.append(np.asarray(psth_sem_truncated[i]) / baseline_activity)
        psth_baseline_std_norm.append(baseline_std / baseline_activity)
        psth_baseline_mean.append(baseline_activity)
    psth_to_be_plotted = np.array(psths_nexus_blzero_norm)
    psth_to_be_plotted_responsive = psth_to_be_plotted.copy()[idx_responsive]
    psth_to_be_plotted_putative_int = psth_to_be_plotted.copy()[idx_putativeint]
    assert all(idx_putativeint_responsive == np.intersect1d(idx_putativeint, idx_responsive))
    psth_to_be_plotted_putative_int_responsive = psth_to_be_plotted.copy()[idx_putativeint_responsive]
    t_psth_to_be_plotted = np.array(psth_t_truncated)
    psths_nexus = psth_truncated  # "raw" psth without normalization
    b_plot_only_putativeint = True
    if b_plot_only_putativeint:
        psth_to_be_plotted = psth_to_be_plotted_putative_int
        psth_to_be_plotted_responsive = psth_to_be_plotted_putative_int_responsive
        idx_responsive = idx_putativeint_responsive

    # region plot example psths of neuronexus recordings as loaded from margarida's script #########################
    marker_colors = [[0.99, 0.11, 0.165], [0.11, 0.44, 0.99], [0.8, 0.366, 0.9]]
    for i, id_nrn in enumerate(id_rec_nrn_nxs):
        ax_psth.append(plt.subplot2grid((n_rows, n_cols), (i_rows_psth[i], i_cols_psth[i]),
                                        colspan=n_cols_psth, rowspan=n_rows_default))
        # plot spike raster
        n_trials = len(spiketimes_nexus[id_nrn - 1])
        dot_color = [.5, .5, .5]
        y_trials = (np.random.permutation(n_trials) + 2) / (n_trials + 4) * y_lim_psths[1]
        for trial in range(n_trials):
            spikes_in_trial = spiketimes_nexus[id_nrn - 1][trial]
            spikes_to_plot = spikes_in_trial[np.logical_and(spikes_in_trial > x_lim_psths[0],
                                                            spikes_in_trial < x_lim_psths[1])]
            ax_psth[i].plot(spikes_to_plot, [y_trials[trial]] * len(spikes_to_plot),
                            '.', markerfacecolor=dot_color, markeredgewidth=0, markersize=2)
        ax_psth_spks.append(ax_psth[i].twinx())
        ax_psth_spks[i].set_ylim([-1, n_trials + 3])
        ax_psth_spks[i].set_yticks(ax_psth_spks[i].get_yticks()[ax_psth_spks[i].get_yticks() > 0])  # no 0 tick
        ax_psth_spks[i].set_ylim([-1, n_trials + 3])
        ax_psth_spks[i].spines['right'].set_color(dot_color)
        ax_psth_spks[i].yaxis.label.set_color(dot_color)
        ax_psth_spks[i].tick_params(axis='y', colors=dot_color)
        ax_psth_spks[i].set_ylabel(' ')
        ax_psth_spks[i].invert_yaxis()
        # call marker and baseline marker
        # ax_psth[i].axvline(0, color=[0, 0, 0], linewidth=1.5, linestyle=':')
        # ax_psth[i].axvline(110, color=[0, 0, 0], linewidth=1.5, linestyle=':')
        ax_psth[i].axhline(1, color=[0, 0, 0], linewidth=1, linestyle='-')
        ax_psth[i].axhline(1 + 2 * psth_baseline_std_norm[id_nrn - 1], color=[0, 0, 0], linewidth=1, linestyle='--')
        ax_psth[i].axhline(1 - 2 * psth_baseline_std_norm[id_nrn - 1], color=[0, 0, 0], linewidth=1, linestyle='--')
        # add box marker for playback time
        box = matplotlib.patches.FancyBboxPatch((0, 0.2), 110, y_lim_psths[1] - 0.4, edgecolor=[0, 0, 0],
                                                fill=False, linewidth=2, linestyle='--')
        box.set_clip_on(False)
        ax_psth[i].add_patch(box)
        # plot psth
        ax_psth[i].plot(t_psth_to_be_plotted[id_nrn - 1], psths_nexus_blnorm[id_nrn - 1], 'k')
        # plot +-sem error margin as patch
        error_path = [psths_nexus_blnorm[id_nrn - 1] - psth_sem_blnorm[id_nrn - 1],
                      psths_nexus_blnorm[id_nrn - 1][::-1] + psth_sem_blnorm[id_nrn - 1][::-1]]
        y_path = np.concatenate([np.concatenate(error_path), np.concatenate(error_path)])
        x_path = np.concatenate([np.arange(0, len(y_path) / 2), np.arange(len(y_path) / 2, 0, -1) - 1]) + \
            t_psth_to_be_plotted[id_nrn - 1][0]
        patch = matplotlib.patches.PathPatch(matplotlib.path.Path(np.transpose([x_path, y_path])),
                                             facecolor=[.5] * 3, edgecolor='none', alpha=.5)
        ax_psth[i].add_patch(patch)
        ax_psth[i].set_xlim(x_lim_psths)
        ax_psth[i].set_ylim(y_lim_psths)
        ax_psth[i].set_ylabel('Spike rate\nrel. to baseline')
        if i == len(id_rec_nrn_nxs) - 1:
            ax_psth[i].set_xlabel('Time from playback onset [ms]')
        else:
            ax_psth[i].set_xlabel('')
            ax_psth[i].set_xticklabels([])
        # add number to identify example psth in matrix plot
        ax_psth[i].plot(450, 2.8, 'o', markersize=12, clip_on=False, markerfacecolor=marker_colors[i], markeredgecolor='k')
        ax_psth[i].text(450, 2.8, str(i + 1), size=FONTSIZE_S, va='center', ha='center')
    ax_psth_spks[0].text(x_lim_psths[1] + 5, 0, 'Trials', ha='left', va='top', size=FONTSIZE_S, color=dot_color)

    # plot psth matrix
    samp_baseline_start = np.where(t_psth_to_be_plotted[0] >= t_baseline_psth_norm[0])[0][0]
    samp_baseline_end = np.where(t_psth_to_be_plotted[0] <= t_baseline_psth_norm[1])[0][-1]
    max_psth = np.max(np.abs(psth_to_be_plotted))
    colormap = plt.cm.get_cmap('RdBu').reversed()
    # get indices to neurons where avg +/- sem crosses above bl + 2*std or below bl - 2*std
    psths_nexus_blnorm_responsive = np.array(psths_nexus_blnorm)[idx_responsive]
    psth_sem_blnorm_responsive = np.array(psth_sem_blnorm)[idx_responsive]
    idx_pos_responsive = []
    idx_neg_responsive = []
    for i, p in enumerate(psths_nexus_blnorm_responsive):
        if any(p - psth_sem_blnorm_responsive[i] > 1 + 2 * psth_baseline_std_norm[idx_responsive[i]]):
            idx_pos_responsive.append(i)
        if any(p + psth_sem_blnorm_responsive[i] < 1 - 2 * psth_baseline_std_norm[idx_responsive[i]]):
            idx_neg_responsive.append(i)
    t_peak = []
    print('--------- responsive neuron ids: \n' + str(idx_responsive[:] + 1))
    for a in range(4):
        ax_mtrx.append(plt.subplot2grid((n_rows, n_cols), (i_rows_mtrx[a], i_cols_mtrx[a]),
                                        colspan=n_cols_mtrx[a], rowspan=n_rows_mtrx[a]))
        if a in [0, 1]:
            psth = psth_to_be_plotted.copy()
        elif a == 2:
            psth = psth_to_be_plotted_responsive.copy()[idx_pos_responsive]
            print('--------- positive responsive neuron ids: \n' + str(idx_responsive[idx_pos_responsive] + 1))
        elif a == 3:
            psth = psth_to_be_plotted_responsive.copy()[idx_neg_responsive]
            print('--------- negative responsive neuron ids: \n' + str(idx_responsive[idx_neg_responsive] + 1))
        # sort by time of peak
        if a in [0, 2]:
            idx_peak = np.argmax(np.array([pp[samp_baseline_end:] for pp in psth]), 1)
        else:
            idx_peak = np.argmin(np.array([pp[samp_baseline_end:] for pp in psth]), 1)
        sort_order = np.argsort(idx_peak)
        if a == 0:
            print('sort order neuron ids for plot ' + str(a) + ': \n')
            print(sort_order + 1)
            [print(str(running + 1) + ': ' + str(i)) for running, i in enumerate(sort_order + 1)]
        if a == 2:
            print('sort order neuron ids for plot ' + str(a) + ': \n' + str(sort_order))
            sort_order_pos = idx_responsive[idx_pos_responsive][sort_order] + 1
            print(idx_responsive[idx_pos_responsive][sort_order] + 1)
            [print(str(running + 1) + ': ' + str(i)) for running, i in
             enumerate(idx_responsive[idx_pos_responsive][sort_order] + 1)]
        elif a == 3:
            print('sort order neuron ids for plot ' + str(a) + ': \n' + str(sort_order))
            sort_order_neg = idx_responsive[idx_neg_responsive][sort_order] + 1
            print(idx_responsive[idx_neg_responsive][sort_order] + 1)
            [print(str(running + 1) + ': ' + str(i)) for running, i in
             enumerate(idx_responsive[idx_neg_responsive][sort_order] + 1)]
        if b_sort_psths:
            psth_sorted = psth.copy()[sort_order]
        else:
            psth_sorted = psth.copy()
        # save peak times
        t_peak.append(t_psth_to_be_plotted[0][idx_peak[sort_order]])
        ax_mtrx[a].matshow(psth_sorted, cmap=colormap, aspect='auto', vmin=-max_psth, vmax=max_psth,
                           extent=[t_psth_to_be_plotted[0][0], t_psth_to_be_plotted[0][-1], len(psth_sorted) + 0.5,
                                   0.5],
                           norm=matplotlib.colors.Normalize(vmin=-max_psth, vmax=max_psth))
        ax_mtrx[a].set_xlabel('Time from playback onset [ms]')
        ax_mtrx[a].set_xlim(x_lim_psths)
        ax_mtrx[a].set_ylim((len(psth_sorted) + 0.5, 0.5))
        ax_mtrx[a].tick_params(axis='x', bottom=True, top=False, labelbottom=True, labeltop=False)
        # add box marker for playback time
        box = matplotlib.patches.FancyBboxPatch((0, 0.5), 110, len(psth_sorted), edgecolor=[0, 0, 0],
                                                fill=False, linewidth=2, linestyle='--')
        box.set_clip_on(False)
        ax_mtrx[a].add_patch(box)
        # add markers for example psths
        for i, id_nrn in enumerate(id_rec_nrn_nxs):
            if a == 2:
                if id_nrn in idx_responsive[idx_pos_responsive] + 1:
                    y_pos = np.where(idx_responsive[idx_pos_responsive][sort_order] + 1 == id_nrn)[0][0] + 1
                    p_plot.arrow(ax_mtrx[a], (-200, y_pos), (-110, y_pos), width=0.07, head_length=12,
                                 head_width_factor=8.0, color=[0, 0, 0], clip_on=False)
                    ax_mtrx[a].plot(-230, y_pos, 'o', markersize=11, clip_on=False, markerfacecolor='w',
                                    markeredgecolor='k')
                    plt.text(-230, y_pos, str(i + 1), size=FONTSIZE_XS, va='center', ha='center')
            elif a == 3:
                if id_nrn in idx_responsive[idx_neg_responsive] + 1:
                    y_pos = np.where(idx_responsive[idx_neg_responsive][sort_order] + 1 == id_nrn)[0][0] + 1
                    p_plot.arrow(ax_mtrx[a], (-200, y_pos), (-110, y_pos), width=0.07, head_length=12,
                                 head_width_factor=8.0, color=[0, 0, 0], clip_on=False)
                    ax_mtrx[a].plot(-230, y_pos, 'o', markersize=11, clip_on=False, markerfacecolor='w',
                                    markeredgecolor='k')
                    plt.text(-230, y_pos, str(i + 1), size=FONTSIZE_XS, va='center', ha='center')
        # adjust y-axis tick positions
        if a in [2, 3]:
            ax_mtrx[a].set_yticks(np.arange(10, len(psth_sorted) + 1, 10))
        ax_mtrx[a].set_ylabel(' ')
    ax_mtrx[2].set_ylabel('Neuron Nr.\n')
    ax_mtrx[0].set_ylabel('Neuron Nr.')
    # add colorbar
    for a in range(4):
        h_cb = plt.colorbar(
            matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=-max_psth, vmax=max_psth),
                                         cmap=colormap),
            ax=ax_mtrx[a], drawedges=False, ticks=np.arange(-1, 1.1, 0.5))
        if a in [1, 3]:
            h_cb.ax.set_ylabel('Normalized deviation from baseline')
        else:
            h_cb.ax.set_ylabel(' ')
        ax_cb.append(h_cb.ax)
    # endregion

    # region baseline activity histogram ################################################################
    ax_hist.append(plt.subplot2grid((n_rows, n_cols), (i_rows_hist[0], i_cols_hist[0]),
                                    colspan=n_cols_hist, rowspan=n_rows_hist))
    ax_hist[0].hist(psth_baseline_mean, 30, color='k')
    ax_hist[0].set_xlim((0, 90))
    ax_hist[0].set_xlabel('Baseline spike rate [Hz]')
    ax_hist[0].set_ylabel('Nr. of neurons')
    # endregion

    # region line plot ################################################################
    ax_line.append(plt.subplot2grid((n_rows, n_cols), (i_rows_line[0], i_cols_line[0]),
                                    colspan=n_cols_line[0], rowspan=n_rows_line[0]))
    psth = psth_to_be_plotted_responsive.copy()
    ax_line[0].set_xlabel('Time')
    ax_line[0].plot((0, x_lim_psths[1]), (1, len(psth)), '-k', linewidth=1)
    ax_line[0].plot(t_peak[3] + 100, np.linspace(1, len(psth), len(t_peak[3])) + 1,
                    color=[0.1, 0.4, 0.9, 2 / 3], linewidth=1.5)
    ax_line[0].plot(t_peak[2] + 100, np.linspace(1, len(psth), len(t_peak[2])) + 1,
                    color=[0.9, 0.1, 0.15, 2 / 3], linewidth=1.5)
    ax_line[0].set_xlim((-50, x_lim_psths[1]))
    ax_line[0].set_ylim((1, len(psth)))
    ax_line[0].invert_yaxis()
    ax_line[0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    # playback marker box
    box = matplotlib.patches.FancyBboxPatch((0, 1), 110, len(psth) - 2, edgecolor=[0, 0, 0],
                                            fill=False, linewidth=0.5, linestyle='--')
    box.set_clip_on(False)
    ax_line[0].add_patch(box)
    ax_line[0].set_ylabel('Neuron Nr.')
    # endregion

    # region horizontal bars ###########################################################################################
    idx_mixed_responsive = np.intersect1d(idx_responsive[idx_neg_responsive], idx_responsive[idx_pos_responsive])
    id_strictly_pos_responsive = sort_order_pos[[sort_order_pos[i] - 1 not in idx_mixed_responsive
                                                 for i in range(len(sort_order_pos))]]
    id_strictly_neg_responsive = sort_order_neg[[sort_order_neg[i] - 1 not in idx_mixed_responsive
                                                 for i in range(len(sort_order_neg))]]
    idx_strictly_pos_to_sorted_responsive = [np.where(sort_order_pos == sp)[0][0] for sp in id_strictly_pos_responsive]
    idx_strictly_neg_to_sorted_responsive = [np.where(sort_order_neg == sn)[0][0] for sn in id_strictly_neg_responsive]
    resp_onsets_strictly_pos_responders = np.array([nexus_responses['on_plus'][i][0]
                                                    for i in range(len(nexus_responses['on_plus']))
                                                    if nexus_responses['sort_order_pos'][i]
                                                    in np.array(idx_strictly_pos_to_sorted_responsive) + 1])
    resp_onsets_strictly_neg_responders = np.array([nexus_responses['on_minus'][i][0]
                                                    for i in range(len(nexus_responses['on_minus']))
                                                    if nexus_responses['sort_order_neg'][i]
                                                    in np.array(idx_strictly_neg_to_sorted_responsive) + 1])
    for a in range(2):
        ax_bar.append(plt.subplot2grid((n_rows, n_cols), (i_rows_bar[a], i_cols_bar[a]),
                                       colspan=n_cols_bar, rowspan=n_rows_bar))
        if a == 0:
            on_cur = nexus_responses['on_plus']
            dur_cur = nexus_responses['dur_plus']
            order_cur = nexus_responses['sort_order_pos']
            color_cur = [0.9, 0.1, 0.15, 0.66]
            print('mean onset time of all ' + str(len(on_cur)) + ' positive responses: ' + str(np.mean(on_cur)))
            print('stdev onset time of all ' + str(len(on_cur)) + ' positive responses: ' + str(np.std(on_cur)))
            print('mean duration of all ' + str(len(dur_cur)) + ' positive responses: ' + str(np.mean(dur_cur)))
            print('stdev duration of all ' + str(len(dur_cur)) + ' positive responses: ' + str(np.std(dur_cur)))
            print('n positive response onsets during playback: ' + str(sum(nexus_responses['on_plus'] <= 110)))
            print('expected number (if response onsets uniformly distr. in time): ' +
                  str(110 / 500 * len(np.unique(nexus_responses['sort_order_pos']))))
            print('n positive response onsets during playback (only strictly positively responding neurons): '
                  + str(sum(resp_onsets_strictly_pos_responders <= 110)))
            print('expected number (if response onsets uniformly distr. in time): ' +
                  str(110 / 500 * len(id_strictly_pos_responsive)))
        else:
            on_cur = nexus_responses['on_minus']
            dur_cur = nexus_responses['dur_minus']
            order_cur = nexus_responses['sort_order_neg']
            color_cur = [0.1, 0.4, 0.9, 0.66]
            print('mean onset time of all ' + str(len(on_cur)) + ' negative responses: ' + str(np.mean(on_cur)))
            print('stdev onset time of all ' + str(len(on_cur)) + ' negative responses: ' + str(np.std(on_cur)))
            print('mean duration of all ' + str(len(dur_cur)) + ' negative responses: ' + str(np.mean(dur_cur)))
            print('stdev duration of all ' + str(len(dur_cur)) + ' negative responses: ' + str(np.std(dur_cur)))
            print('n negative response onsets during playback: ' + str(sum(nexus_responses['on_minus'] <= 110)))
            print('expected number (if response onsets uniformly distr. in time): ' +
                  str(110 / 500 * len(np.unique(nexus_responses['sort_order_neg']))))
            print('n negative response onsets during playback (only strictly negatively responding neurons): '
                  + str(sum(resp_onsets_strictly_neg_responders <= 110)))
            print('expected number (if response onsets uniformly distr. in time): ' +
                  str(110 / 500 * len(id_strictly_neg_responsive)))
        for b in range(len(on_cur)):
            hor_bar = matplotlib.patches.Rectangle((on_cur[b][0], order_cur[b][0] - 0.3), dur_cur[b][0], 0.6,
                                                   color=color_cur, fill=True, clip_on=True, linewidth=0)
            ax_bar[a].add_patch(hor_bar)
        ax_bar[a].set_xlim((-100, 500))
        ax_bar[a].set_ylim((0.1, max(order_cur) + 0.9))
        ax_bar[a].invert_yaxis()
        # ax_bar[a].get_yaxis().set_visible(False)
        # ax_bar[a].spines['left'].set_visible(False)
        ax_bar[a].set_ylabel('Neuron Nr.')
        ax_bar[a].set_xlabel('')
        ax_bar[a].set_xticklabels([])
        # add box marker for playback time
        box = matplotlib.patches.FancyBboxPatch((0, 0), 110, max(order_cur) + 1, edgecolor=[0, 0, 0],
                                                fill=False, linewidth=2, linestyle='--', clip_on=False)
        ax_bar[a].add_patch(box)
    # plot sum of bars
    for a in range(2):
        ax_sum.append(plt.subplot2grid((n_rows, n_cols), (i_rows_sum[a], i_cols_sum[a]),
                                       colspan=n_cols_sum, rowspan=n_rows_sum))
        if a == 0:
            sum_cur = nexus_responses['sum_pos']
            color_cur = [0.9, 0.1, 0.15, 0.66]
        else:
            sum_cur = nexus_responses['sum_neg']
            color_cur = [0.1, 0.4, 0.9, 0.66]
        ax_sum[a].plot(range(500), (sum_cur / nexus_responses['putativeint_eligible_neurons'])[0], color=color_cur)
        ax_sum[a].set_xlabel('Time from playback onset [ms]')
        ax_sum[a].set_ylabel(' ')
        ax_sum[a].set_xlim((-100, 500))
        ax_sum[a].set_ylim((0, 0.15))
    # endregion

    # add sub-figure labels
    ax_psth[0].annotate('A', xy=(-0.95 / n_cols_psth, 1.11), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_hist[0].annotate('B', xy=(-0.65 / n_cols_hist, 1.11), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_mtrx[0].annotate('C', xy=(-1.19 / n_cols_mtrx[0], 1.03), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_mtrx[2].annotate('D', xy=(-1.19 / n_cols_mtrx[2], 1.05), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_line[0].annotate('F', xy=(-0.5 / n_cols_line[0], 1.11), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_bar[0].annotate('E', xy=(-1.19 / n_cols_bar, 1.05), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    # adjust position of horizontal bar and sum plots
    for a in range(2):
        ax_pos = ax_bar[a].get_position()
        ax_pos_new = [ax_pos.x0, ax_pos.y0, ax_pos.width - 0.02, ax_pos.height]
        ax_bar[a].set_position(ax_pos_new)
        ax_pos = ax_sum[a].get_position()
        ax_pos_new = [ax_pos.x0, ax_pos.y0, ax_pos.width - 0.021, ax_pos.height]
        ax_sum[a].set_position(ax_pos_new)

    # move line plot to the left
        ax_pos = ax_line[0].get_position()
        ax_pos_new = [ax_pos.x0 - 0.01, ax_pos.y0, ax_pos.width, ax_pos.height]
        ax_line[0].set_position(ax_pos_new)

    return fig, ax_psth + ax_psth_spks + ax_mtrx + ax_cb + ax_line + ax_hist + ax_bar + ax_sum


def figure4(spikes_jam_only, info_jam_only, config_jam_only,
            states_agg_jam_only, info_agg_jam_only, spiketimes_nexus, traces_playback_aligned,
            traces_nif, spiketimes_nif, suppr_win, img_zf_listening, img_bubble, img_speaker):
    # figure parameters
    n_rows = 15
    n_cols = 19
    i_rows_call = [5]  # call playback marker
    i_cols_call = [0]
    i_rows_nif_trc = [7]  # traces of NIf cells
    i_cols_nif_trc = [0]
    n_rows_nif_trc = 2
    n_cols_nif_trc = 4
    i_rows_nif = [9]  # psths of NIf cells
    i_cols_nif = [0]
    n_rows_nif = 2
    n_cols_nif = 4
    i_rows_crc = [1]  # circuit diagram
    i_cols_crc = [0]
    n_rows_crc = 4
    n_cols_crc = 4
    i_rows_psth = [1, 3]  # psths of Aud and Jam
    i_cols_psth = [5, 5]
    n_cols_psth = 4
    i_rows_agg = [6]  # aggregate trace and average of jam-only sim
    i_cols_agg = [5]
    n_rows_agg = 2
    n_cols_agg = 4
    i_rows_sub = [9]  # playback-aligned subthreshold potentials of bursting PM
    i_cols_sub = [5]
    n_rows_sub = 2
    n_cols_sub = 4
    n_rows_default = 2
    x_lim_psths = [-100, 125]
    y_lim_psths_nif = [-3, 20]
    x_lim_agg = [-100, 125]
    color_map_rec = p_plot.get_color_list(8, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])
    color_map_mod = p_plot.get_color_list(8, color_map_in=[C_COLORS[c] for c in COL_SEQ])

    # file, population & trace indices
    playback_onset_mod_jam_only = config_jam_only['misc']['playback_start']
    call_onset_ms_playback_aligned = 200
    t_baseline_psth_norm = (-100, 0)  # for normalizing psths to baseline. Set to None to not normalize
    id_rec_nrn_nxs = [2, 5, 8, 11, 13, 14, 15, 16, 17, 20, 21, 39, 41, 42, 54, 55, 62, 64, 73, 94, 148, 151, 155,
                      161, 165, 168, 169]  # final (putative interneurons with a significant activity peak w/in 100ms)
    letter_pop_mod = ['V', '$\mathbf{I_v}$', 'P', '$\mathbf{I_a}$', 'A']

    # set up figure
    fig = plt.figure(4, figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_crc = []
    ax_nif = []
    ax_nif_spks = []
    ax_nif_trc = []
    ax_psth = []
    ax_agg = []
    ax_sub = []
    ax_lat = []
    ax_trc = []

    # region call-marker axis ###################################################################################
    y_lim_call = (0, 2)
    ax_call = plt.subplot2grid((n_rows, n_cols), (i_rows_call[0], i_cols_call[0]),
                               colspan=n_cols_psth, rowspan=n_rows_default)
    ax_call.plot((0, 0), (0, 1), ':k', linewidth=1)
    ax_call.plot((110, 110), (0, 1), ':k', linewidth=1)
    x = np.arange(0, 110, 0.1)
    sine = np.sin(x * 0.75)
    gauss = norm.pdf(x, 55, 22) * 55 - 0.05
    wave = (sine * gauss / 2.1 + 0.5)
    # embed zebra finch image
    ax_call_bbox = ax_call.get_window_extent()
    ax_call_axes_ratio = ax_call_bbox.width / ax_call_bbox.height
    ax_call_data_ratio = (x_lim_psths[1] - x_lim_psths[0]) / (y_lim_call[1] - y_lim_call[0])
    inset_height = 1.2
    correction_factor = 0.888888  # apparrently axis gets stretched in finale saved image. this corrects for that
    ax_call.imshow(img_zf_listening, aspect='auto', interpolation='none', zorder=2,
                   extent=(-52 + correction_factor * inset_height * ax_call_data_ratio / ax_call_axes_ratio, -52, 0,
                           inset_height))
    ax_inset_img = ax_call.inset_axes((x_lim_psths[0], 0.9, 45, 0.9), transform=ax_call.transData, zorder=1.5)
    ax_inset_img.imshow(img_bubble, aspect='auto', interpolation='none', extent=(-10, 120, -0.8, 1.50), clip_on=False)
    ax_inset_img.plot(x, wave, '-k', linewidth=0.5)
    ax_inset_img.axis('off')
    speaker_height = 0.7
    ax_call.imshow(img_speaker, aspect='auto', interpolation='none', clip_on=False,
                   extent=(-130, -130 + correction_factor * speaker_height * ax_call_data_ratio / ax_call_axes_ratio,
                           0.2, 0.2 + speaker_height))
    # plot call marker
    ax_call.plot(x, wave, '-k', linewidth=1)
    ax_call.text(50, 1.5, 'Call playback', va='center', ha='center')
    ax_call.set_xlim(x_lim_psths)
    ax_call.set_ylim(y_lim_call)
    ax_call.axis('off')
    # plot fake x-axis from call onset onwards
    ax_call.plot((0, 110), (y_lim_call[0], y_lim_call[0]), '-k', clip_on=False, linewidth=0.8)
    for xt in [0, 50, 100]:
        ax_call.plot((xt, xt), (y_lim_call[0], y_lim_call[0] - 0.1), '-k', clip_on=False, linewidth=0.8)
    # endregion

    # region plot circuit diagram ##############################################################################
    x_circuit = [[2, 2, 2, 3, 4, 4, 4]]
    y_circuit = [[3, 2, 1, 1, 1, 2, 3]]
    conn_circ = [[(0, 1), (1, 2), (1, 3), (2, 3), (4, 3), (5, 3), (5, 4), (6, 5)]]
    conn_type = [[1.5, 1.5, 1.5, -1.5, -1.5, 1.5, 1.5, 1.5]]
    pop_gray = [.8, .8, .8]
    color_circ = [[pop_gray, pop_gray, pop_gray, color_map_mod[3], color_map_mod[7], color_map_mod[6], [0, 0, 0]]]
    letters = [[None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2],
                letter_pop_mod[3], letter_pop_mod[4], None]]
    color_conn = [[pop_gray, pop_gray, pop_gray, pop_gray, [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    ramp_current = np.concatenate((np.zeros(5), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 70 + 1)],
                                   np.linspace(0.1, 0, 10), np.zeros(20)))
    pulse_current = np.concatenate((np.zeros(33), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 20)],
                                    np.linspace(0.1, 0, 20), np.zeros(33)))
    currents = [[ramp_current] + [None] * (len(x_circuit[0]) - 2) + [pulse_current]]
    rectangles = [[2, 1, 2, 0]]
    # set axis limits and plot
    x_lim_crc = (min([min(v) for v in x_circuit]) + 0.3, max([max(v) for v in x_circuit]) + 0.3)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.9, max([max(v) for v in y_circuit]) + 0.4)
    for i in range(len(x_circuit)):
        ax_crc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc[i], i_cols_crc[i]),
                                       colspan=n_cols_crc, rowspan=n_rows_crc))
        circuit_diagram(ax_crc[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc, region_label='HVC',
                        connection_colors=color_conn[i], letter_size=15, region_rectangle_xywh=rectangles[i])
    # add text describing inputs next to circuit diagram
    ax_crc[0].text(1.05, 3, 'vocal-\nrelated\ninput', va='center', ha='center',
                   fontsize=11, fontweight='bold', color=[.8, .8, .8])
    ax_crc[0].text(4.95, 3, 'auditory-\nrelated\ninput', va='center', ha='center',
                   fontsize=11, fontweight='bold', color='k')
    # endregion

    # region plot traces of NIf cells ################################################################
    playback_onset_nif = [500, 300]
    idx_nif_cells_to_plot = [1]
    fs_nif_khz = 40
    idx_trace = 66  # 4, 69, 78
    for i, idx_nif in enumerate(idx_nif_cells_to_plot):
        ax_nif_trc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_nif_trc[i], i_cols_nif_trc[i]),
                                           colspan=n_cols_nif_trc, rowspan=n_rows_nif_trc))
        trace_nif_raw = traces_nif[idx_nif][idx_trace]
        t_trc_nif = np.linspace(0, len(trace_nif_raw) / fs_nif_khz, len(trace_nif_raw)) - playback_onset_nif[idx_nif]
        trace_nif_cut = trace_nif_raw[np.logical_and(t_trc_nif > x_lim_psths[0], t_trc_nif < x_lim_psths[1])]
        t_trc_nif_cut = t_trc_nif[np.logical_and(t_trc_nif > x_lim_psths[0], t_trc_nif < x_lim_psths[1])]
        ax_nif_trc[i].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
        ax_nif_trc[i].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')
        ax_nif_trc[i].plot(t_trc_nif_cut, trace_nif_cut, '-k', linewidth=1)
        ax_nif_trc[i].set_xlim(x_lim_psths)
        ax_nif_trc[i].set_xlabel('')
        ax_nif_trc[i].set_xticklabels([])
        ax_nif_trc[i].set_ylabel('$\mathrm{V_m}$ [mV]')
    ax_nif_trc[0].text(0.05, 0.8, '$\mathrm{NIf_{HVC}}$', size=FONTSIZE_S, color='k', transform=ax_nif_trc[0].transAxes)
    # endregion

    # region plot psth of NIf cells ################################################################
    for i, idx_nif in enumerate(idx_nif_cells_to_plot):
        ax_nif.append(plt.subplot2grid((n_rows, n_cols), (i_rows_nif[i], i_cols_nif[i]),
                                       colspan=n_cols_nif, rowspan=n_rows_nif))
        # plot spike raster
        n_trials = len(spiketimes_nif[idx_nif])
        dot_color = [.5, .5, .5]
        for trial in range(n_trials):
            spikes_in_trial = np.array(spiketimes_nif[idx_nif][trial]) - playback_onset_nif[idx_nif]
            spikes_to_plot = spikes_in_trial[np.logical_and(spikes_in_trial > x_lim_psths[0],
                                                            spikes_in_trial < x_lim_psths[1])]
            ax_nif[i].plot(spikes_to_plot, [trial / n_trials * (y_lim_psths_nif[1] - y_lim_psths_nif[0])
                                            + y_lim_psths_nif[0]] * len(spikes_to_plot),
                           '.', markerfacecolor=dot_color, markeredgewidth=0, markersize=4)
        ax_nif_spks.append(ax_nif[i].twinx())
        ax_nif_spks[i].set_ylim([0, n_trials])
        ax_nif_spks[i].set_yticks(ax_nif_spks[i].get_yticks()[ax_nif_spks[i].get_yticks() > 0])  # no 0 tick
        ax_nif_spks[i].set_ylim([0, n_trials])
        ax_nif_spks[i].spines['right'].set_color(dot_color)
        ax_nif_spks[i].yaxis.label.set_color(dot_color)
        ax_nif_spks[i].tick_params(axis='y', colors=dot_color)
        ax_nif_spks[i].set_ylabel(' ')
        ax_nif_spks[i].text(x_lim_psths[1], 0, 'Trials', ha='left', va='top', size=FONTSIZE_S, color=dot_color)
        ax_nif_spks[i].invert_yaxis()
        # plot psth
        t_nif, _, psth_nif_smooth = p_analysis.get_psth(spiketimes_nif[idx_nif], info_jam_only[0]['sim_time'])
        t_nif = [[v - playback_onset_nif[idx_nif] for v in t_nif]]
        ax_nif[i].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
        p_plot.plot_psths(t_nif, psths_smooth=psth_nif_smooth, t_const_event_marker=0, h_ax=ax_nif[i],
                          colors=['k'], marker_linestyle=':')
        ax_nif[i].get_legend().remove()
        ax_nif[i].set_title('')
        ax_nif[i].set_xlim(x_lim_psths)
        ax_nif[i].set_ylim(y_lim_psths_nif)
        if i == len(idx_nif_cells_to_plot) - 1:
            ax_nif[i].set_xlabel('Time from playback onset [ms]')
            ax_nif[i].set_ylabel(' ')
        else:
            ax_nif[i].set_xlabel('')
            ax_nif[i].set_xticklabels([])
    ax_nif[0].set_ylabel('Spike rate [Hz]')
    if len(idx_nif_cells_to_plot) == 2:
        ax_nif[0].yaxis.set_label_coords(-0.13, -0.4)
    # endregion

    # region plot psth of auditory population (that drives jamming interneurons) ####################
    ax_psth.append(plt.subplot2grid((n_rows, n_cols), (i_rows_psth[0], i_cols_psth[0]),
                                    colspan=n_cols_psth, rowspan=n_rows_default))
    spiketimes_aud = p_util.get_spiketimes_from_monitor(spikes_jam_only[0], info_jam_only[0], 4)
    t_aud_mod, _, psth_aud_mod_smooth = p_analysis.get_psth(spiketimes_aud, info_jam_only[0]['sim_time'])
    t_aud_mod = [[v - playback_onset_mod_jam_only for v in t_aud_mod]]
    ax_psth[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
    p_plot.plot_psths(t_aud_mod, psths_smooth=psth_aud_mod_smooth, t_const_event_marker=0, h_ax=ax_psth[0],
                      colors=[color_map_mod[6]], marker_linestyle=':', t_baseline=t_baseline_psth_norm,
                      b_normalize_rate=True, labels=['model'])
    ax_psth[0].set_ylabel('Spike rate\nrel. to baseline')
    # ax_psth[0].get_legend().remove()
    ax_psth[0].set_xticklabels([])
    ax_psth[0].set_xlabel('')
    ax_psth[0].legend(loc='upper left')
    # inset axis with population circle
    y_lim_psth = ax_psth[0].get_ylim()
    ax_inset_pop = ax_psth[0].inset_axes((x_lim_psths[0] + 188, y_lim_psth[1] / 2, 50, y_lim_psth[1] / 1.5),
                                         transform=ax_psth[0].transData)
    circuit_diagram(ax_inset_pop, [1], [2], [color_map_mod[6]], [], [], letters=['A'], letter_size=13)
    ax_inset_pop.axis('off')
    # endregion

    # region plot psth of jamming interneurons with comparison to neuronexus recording ##########################
    ax_psth.append(plt.subplot2grid((n_rows, n_cols), (i_rows_psth[1], i_cols_psth[1]),
                                    colspan=n_cols_psth, rowspan=n_rows_default))
    spiketimes_int = p_util.get_spiketimes_from_monitor(spikes_jam_only[0], info_jam_only[0], 3)
    t_int_mod, _, psth_int_mod_smooth = p_analysis.get_psth(spiketimes_int, info_jam_only[0]['sim_time'])
    t_int_rec_offset = []
    psth_int_rec_smooth = []
    for id_nrn in id_rec_nrn_nxs:
        # find the index to spiketimes_nexus that points to the correct unit-playback combination
        t_tmp, _, psth_tmp_smooth = p_analysis.get_psth(spiketimes_nexus[id_nrn - 1] + 200,
                                                        # note: adding 200 = quick fix (also trim_monitors_to_..?)
                                                        info_jam_only[0]['sim_time'])
        t_int_rec_offset.append([v - 200 for v in t_tmp])
        psth_int_rec_smooth.append(psth_tmp_smooth)
    t_int_all = t_int_rec_offset + [[v - playback_onset_mod_jam_only for v in t_int_mod]]
    psth_int_smooth_all = psth_int_rec_smooth + [psth_int_mod_smooth]
    colors_psth = [color_map_rec[7]] * len(psth_int_rec_smooth) + [color_map_mod[7]]
    labels_psth = ['observed'] + [None] * (len(id_rec_nrn_nxs) - 1) + ['model']
    linewidth_psth = [1] * len(t_int_rec_offset) + [2]
    ax_psth[1].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
    p_plot.plot_psths(t_int_all, psths_smooth=psth_int_smooth_all, t_const_event_marker=0, h_ax=ax_psth[1],
                      colors=colors_psth, labels=labels_psth, marker_linestyle=':', linewidth=linewidth_psth,
                      t_baseline=t_baseline_psth_norm, b_legend=True, b_normalize_rate=True)
    ax_psth[1].set_ylabel('Spike rate\nrel. to baseline')
    ax_psth[1].set_xlabel('Time from playback onset [ms]')
    ax_psth[0].legend(loc='upper left')
    for i in range(len(ax_psth)):
        ax_psth[i].set_title('')
        ax_psth[i].set_xlim(x_lim_psths)
    # inset axis with population circle
    y_lim_psth = ax_psth[1].get_ylim()
    ax_inset_pop = ax_psth[1].inset_axes((x_lim_psths[0] + 188, y_lim_psth[1] / 2, 50, y_lim_psth[1] / 1.5),
                                         transform=ax_psth[1].transData)
    circuit_diagram(ax_inset_pop, [1], [2], [color_map_mod[7]], [], [], letters=['$\mathbf{I_a}$'], letter_size=13)
    ax_inset_pop.axis('off')
    # endregion

    # region plot aggregate traces of jam only sim ##############################################################
    ax_agg.append(plt.subplot2grid((n_rows, n_cols), (i_rows_agg[0], i_cols_agg[0]),
                                   colspan=n_cols_agg, rowspan=n_rows_agg))
    trim_time_span = (x_lim_agg[0] + playback_onset_mod_jam_only, x_lim_agg[1] + playback_onset_mod_jam_only)
    states_agg_jam_only_trim, _ = p_util.trim_monitors_to_time_span(trim_time_span, statemons=states_agg_jam_only)
    ax_agg[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':', zorder=2.5)  # playback offset
    p_plot.plot_traces_mod(states_agg_jam_only_trim, info_agg_jam_only, t_offset_ms=playback_onset_mod_jam_only,
                           b_spike_threshold=False, b_average_aggregate=True, t_const_event_marker=0,
                           trace_color=[.85, .85, .85], average_color=color_map_mod[3], h_ax=ax_agg[0],
                           marker_linestyle=':')
    ax_agg[0].set_xlabel('Time from playback onset [ms]')
    ax_agg[0].set_ylabel('$\mathrm{V_m}$ [mV]')
    ax_agg[0].set_xlim(x_lim_agg)
    ax_agg[0].legend(loc='lower left')
    # endregion

    # region playback-aligned observed PM
    ax_sub.append(plt.subplot2grid((n_rows, n_cols), (i_rows_sub[0], i_cols_sub[0]),
                                   colspan=n_cols_sub, rowspan=n_rows_sub))
    samp_freq_khz = 40
    downsampling_factor = 20
    trace_playback_aligned_trim = p_util.trim_traces_to_time_span([call_onset_ms_playback_aligned - 100,
                                                                   call_onset_ms_playback_aligned + 125],
                                                                  traces_playback_aligned[1], samp_freq_khz)
    trace_playback_aligned_trim_down = p_util.downsample_trace(trace_playback_aligned_trim, downsampling_factor)
    # DE-MEAN
    for i in range(len(trace_playback_aligned_trim_down)):
        trace_playback_aligned_trim_down[i] = trace_playback_aligned_trim_down[i] \
                                              - np.mean(trace_playback_aligned_trim_down[i])
    ax_sub[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':', zorder=2.5)  # playback offset
    p_plot.plot_traces_rec(trace_playback_aligned_trim_down, sampling_frequency_khz=samp_freq_khz / downsampling_factor,
                           b_plot_raw_traces=False, b_plot_smoothed=True, b_plot_average=True, b_std_interval=False,
                           b_offset_by_average=False, b_n_in_legend=False, t_const_event_marker=0, marker_linestyle=':',
                           trace_color_smooth=[.85, .85, .85], average_color=color_map_rec[3], t_offset_ms=100,
                           b_average_mean_2std=True, t_baseline_ms=[0, 100], h_ax=ax_sub[0])
    h_l = ax_sub[0].legend(loc='lower left')
    h_l.get_texts()[0].set_text('observed')
    h_l.get_texts()[1].set_text('average')
    ax_sub[0].set_xlabel('Time from playback onset [ms]')
    ax_sub[0].set_ylabel('$\mathrm{V_m}$ [mV]')
    ax_sub[0].set_xlim(x_lim_agg)
    ax_sub[0].set_ylim((-6, 4))
    # endregion

    # add sub-figure labels
    ax_call.annotate('B', xy=(-0.9 / n_cols_psth, 0.9), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc[0].annotate('A', xy=(-0.98, 1.03), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psth[0].annotate('C', xy=(-0.83 / n_cols_psth, 1.08), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_agg[0].annotate('D', xy=(-0.83 / n_cols_agg, 1.08), xycoords='axes fraction', fontweight='bold',
                       size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    return fig, ax_crc + [ax_call] + ax_psth + ax_agg + ax_sub + ax_nif + ax_nif_spks + ax_nif_trc


def figure5(states_jam_only, spikes_jam_only, info_jam_only, config_jam_only,
            states_jam_ft, spikes_jam_ft, config_jam_ft, info_jam_ft, spiketimes_spiking, spiketimes_gabazine,
            call_onset_times, suppr_win):
    # figure parameters
    n_rows = 25
    n_cols = 25
    i_rows_tpr = [9]  # trace-per-run
    i_cols_tpr = [5]
    n_cols_tpr = 4
    n_rows_tpr = 3
    i_rows_crc = [0]  # circuit diagram
    i_cols_crc = [0]
    n_rows_crc = 4
    n_cols_crc = 4
    i_rows_lat = [4]
    i_cols_lat = [0]
    n_rows_lat = 1
    n_cols_lat = 4
    i_rows_trc = [6, 8, 10]  # example traces of bursting PM at different call onsets relative to playback
    i_cols_trc = [0, 0, 0]
    n_rows_trc = 2
    n_cols_trc = 4
    i_rows_brst = [0]  # burst onsets of all recorded bursting PMs
    i_cols_brst = [5]
    n_rows_brst = 4
    n_cols_brst = 4
    i_rows_ovrlp = [0, 2]  # burst onsets of all recorded bursting PMs
    i_cols_ovrlp = [10, 10]
    n_rows_ovrlp = [2, 1]
    n_cols_ovrlp = 3
    i_rows_hist = [4, 6, 8, 10]  # histograms of call onsets (excerpt; sim, control, gabazine)
    i_cols_hist = [10, 10, 10, 10]
    n_rows_hist = 2
    n_cols_hist = 3
    i_rows_hist_all = [5]  # histograms of call onsets (full range; control, gabazine)
    i_cols_hist_all = [5]
    n_rows_hist_all = 3
    n_cols_hist_all = 4
    x_lim_tpr = [-100, 125]
    x_lim_traces = [-25, 135]
    y_lim_traces = [(-70, -5)] * 3
    linewidth = 2
    color_map_rec = p_plot.get_color_list(8, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])
    color_map_mod = p_plot.get_color_list(8, color_map_in=[C_COLORS[c] for c in COL_SEQ])

    # file, population & trace indices etc.
    idx_nrn_oi_mod_abs_tpr = p_util.get_abs_from_rel_nrn_idx(config_jam_only['plot']['idx_nrn_oi_relative'],
                                                             config_jam_only['plot']['idx_pop_oi'],
                                                             info_jam_only[0]['population_sizes'])
    playback_onset_mod_jam_only = config_jam_only['misc']['playback_start']
    rec_offset_ms_spiking = [[100] * 12, [100, 100, 100, 100, 100, 200, 100, 100]]  # realigned (last 100 before)
    idx_mon_oi_mod = [22, 17, 11]
    idx_nrn_oi_mod_abs_trc = p_util.get_abs_from_rel_nrn_idx(config_jam_ft['plot']['idx_nrn_oi_relative'],
                                                             config_jam_ft['plot']['idx_pop_oi'],
                                                             info_jam_ft[0]['population_sizes'])
    idx_nrn_oi_mod_abs_tpr = p_util.get_abs_from_rel_nrn_idx(config_jam_only['plot']['idx_nrn_oi_relative'],
                                                             config_jam_only['plot']['idx_pop_oi'],
                                                             info_jam_only[0]['population_sizes'])
    letter_pop_mod = ['V', '$\mathbf{I_v}$', 'P', '$\mathbf{I_a}$', 'A']
    artificial_spike_heights = 25

    # set up figure
    fig = plt.figure(5, figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    xlims_burst = (0 - rec_offset_ms_spiking[0][0], 160 - rec_offset_ms_spiking[0][0])
    ax_tpr = []
    ax_crc = []
    ax_trc = []
    ax_lat = []
    ax_brst = []
    ax_ovrlp = []
    ax_hist = []
    ax_hist_all = []

    # region plot trace per run jam only ########################################################################
    # add artificial spikes by increasing the potential value at spiketimes by a predefined amount
    if artificial_spike_heights:
        for run in range(len(spikes_jam_only)):
            spiketimes_tpr = spikes_jam_only[run].t[spikes_jam_only[run].i == idx_nrn_oi_mod_abs_tpr] / b2.ms
            for spk in spiketimes_tpr:
                samp_spike = int(round(spk / (info_jam_only[0]['dt'] / b2.ms)))
                states_jam_only[run].v[idx_nrn_oi_mod_abs_tpr, samp_spike] = \
                    states_jam_only[run].v[idx_nrn_oi_mod_abs_tpr, samp_spike] + artificial_spike_heights * b2.mV
    # trim monitors to time span in plot
    trim_time_span = (x_lim_tpr[0] + playback_onset_mod_jam_only, x_lim_tpr[1] + playback_onset_mod_jam_only)
    states_jam_only_trim, _ = p_util.trim_monitors_to_time_span(trim_time_span, statemons=states_jam_only)
    # plot traces
    ax_tpr.append(plt.subplot2grid((n_rows, n_cols), (i_rows_tpr[0], i_cols_tpr[0]),
                                   colspan=n_cols_tpr, rowspan=n_rows_tpr))
    ax_tpr[0].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
    ax_tpr[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')
    p_plot.plot_traces_per_run(states_jam_only_trim, config_jam_only, info_jam_only,
                               t_offset_ms=playback_onset_mod_jam_only,
                               b_spike_threshold=False, neuron_idx=0, population_idx=2, b_maxmin_marker=False,
                               b_legend=False, h_ax=ax_tpr[0])
    # format axis
    ax_tpr[0].tick_params(labelright=False)
    ax_tpr[0].set_xlim(x_lim_tpr)
    ymin = min(states_jam_only_trim[0].v[idx_nrn_oi_mod_abs_tpr, :] / b2.mV) - 5
    ax_tpr[0].set_ylim((ymin, -10))
    ax_tpr[0].set_ylabel('$\mathrm{V_m}$ [mV]')
    ax_tpr[0].set_xlabel('Time from playback onset [ms]')
    ax_tpr[0].set_title('Model premotor neuron', fontsize=11, fontweight='bold')
    # add colorbar to axis (multiply by 1000 to get pA instead of nA)
    min_weight = info_jam_only[0]['free_parameter_values'][config_jam_only['plot']['idx_synpop_oi_for_fp']] * 1000
    max_weight = info_jam_only[-1]['free_parameter_values'][config_jam_only['plot']['idx_synpop_oi_for_fp']] * 1000
    h_cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=min_weight, vmax=max_weight),
                                                     cmap='viridis'), ax=ax_tpr[0], drawedges=False,
                        ticks=[-5, -10, -15, -20, -25])
    h_cb.ax.set_ylabel('Inhibitory weights [pA]')
    # endregion

    # region plot circuit diagram ##############################################################################
    x_circuit = [[2, 2, 2, 3, 4, 4, 4]]
    y_circuit = [[3, 2, 1, 1, 1, 2, 3]]
    conn_circ = [[(0, 1), (1, 2), (1, 3), (2, 3), (4, 3), (5, 3), (5, 4), (6, 5)]]
    conn_type = [[1.5, 1.5, 1.5, -1.5, -1.5, 1.5, 1.5, 1.5]]
    color_circ = [[[0, 0, 0], color_map_mod[0], color_map_mod[1], color_map_mod[3],
                   color_map_mod[7], color_map_mod[6], [0, 0, 0]]]
    letters = [[None, letter_pop_mod[0], letter_pop_mod[1], letter_pop_mod[2],
                letter_pop_mod[3], letter_pop_mod[4], None]]
    ramp_current = np.concatenate((np.zeros(5), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 70 + 1)],
                                   np.linspace(0.1, 0, 10), np.zeros(20)))
    pulse_current = np.concatenate((np.zeros(33), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 20)],
                                    np.linspace(0.1, 0, 20), np.zeros(33)))
    currents = [[ramp_current] + [None] * (len(x_circuit[0]) - 2) + [pulse_current]]
    rectangles = [[2, 1, 2, 0]]
    # set axis limits and plot
    x_lim_crc = (min([min(v) for v in x_circuit]) + 0.5, max([max(v) for v in x_circuit]) + 0.5)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.9, max([max(v) for v in y_circuit]) + 0.4)
    for i in range(len(x_circuit)):
        ax_crc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc[i], i_cols_crc[i]),
                                       colspan=n_cols_crc, rowspan=n_rows_crc))
        circuit_diagram(ax_crc[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc, region_label='HVC',
                        letter_size=15, region_rectangle_xywh=rectangles[i])
    # add text describing inputs to circuit diagram
    ax_crc[0].text(1.05, 3, 'vocal-\nrelated\ninput', va='center', ha='center',
                   fontsize=11, fontweight='bold', color='k')
    ax_crc[0].text(4.95, 3, 'auditory-\nrelated\ninput', va='center', ha='center',
                   fontsize=11, fontweight='bold', color='k')
    # endregion

    # region plot time of first spike (latency) versus time of jamming inhibition ###############################
    ax_lat.append(plt.subplot2grid((n_rows, n_cols), (i_rows_lat[0], i_cols_lat[0]),
                                   colspan=n_cols_lat, rowspan=n_rows_lat))
    idx_t_start = 1
    idx_pop_fp = config_jam_only['plot']['idx_synpop_oi_for_fp']
    lat_playback_to_t_start = config_jam_only['input_current']['t_start'][idx_pop_fp][idx_t_start] \
                              - config_jam_only['misc']['playback_start']
    t_first_spike_no_jam = spikes_jam_ft[-1].t[np.where(spikes_jam_ft[-1].i == idx_nrn_oi_mod_abs_trc)[0][0]] / b2.ms
    print('DEBUG: lat_playback_onset_to_t_start = ' + str(lat_playback_to_t_start))
    ax_lat[0].axvspan(suppr_win[0], suppr_win[1], facecolor=[.8, .8, .8], alpha=0.5)
    ax_lat[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
    p_plot.plot_t_first_spike_1d_jamming(spikes_jam_ft, config_jam_ft, info_jam_ft, idx_nrn_oi_mod_abs_tpr,
                                         idx_pop_fp, idx_t_start=idx_t_start, marker_linestyle=':',
                                         lat_playback_onset_to_t_start=lat_playback_to_t_start,
                                         dot_size=3, h_ax=ax_lat[0], idx_mons_colored=idx_mon_oi_mod,
                                         colors=[color_map_rec[3]] * len(idx_mon_oi_mod))
    # t_const_event_marker=None)
    ax_lat[0].set_title('')
    ax_lat[0].set_xlim(x_lim_traces)
    ax_lat[0].set_ylim((-1, 6))
    ax_lat[0].set_xlabel('Time of unperturbed burst onset\nfrom playback onset [ms]', weight='normal')
    ax_lat[0].set_ylabel('Model burst\ndelay [ms]', weight='normal')
    # endregion

    # region plot example traces of bursting PM at different times relative to jamming interneuron, i.e. playback
    traces_ft, samp_freq_ft = p_util.get_traces_from_monitors(states_jam_ft, b_keep_unit=False)
    traces = [traces_ft[idx_mon_oi_mod[0]][idx_nrn_oi_mod_abs_trc],
              traces_ft[idx_mon_oi_mod[1]][idx_nrn_oi_mod_abs_trc],
              traces_ft[idx_mon_oi_mod[2]][idx_nrn_oi_mod_abs_trc]]
    spiketimes_trc = [
        spikes_jam_ft[idx_mon_oi_mod[0]].t[spikes_jam_ft[idx_mon_oi_mod[0]].i == idx_nrn_oi_mod_abs_trc] / b2.ms,
        spikes_jam_ft[idx_mon_oi_mod[1]].t[spikes_jam_ft[idx_mon_oi_mod[1]].i == idx_nrn_oi_mod_abs_trc] / b2.ms,
        spikes_jam_ft[idx_mon_oi_mod[2]].t[spikes_jam_ft[idx_mon_oi_mod[2]].i == idx_nrn_oi_mod_abs_trc] / b2.ms]
    samp_freqs = [samp_freq_ft, samp_freq_ft, samp_freq_ft]
    offset_ms = [
        info_jam_ft[idx_mon_oi_mod[0]]['free_parameter_values'][idx_pop_fp][idx_t_start] - lat_playback_to_t_start,
        info_jam_ft[idx_mon_oi_mod[1]]['free_parameter_values'][idx_pop_fp][idx_t_start] - lat_playback_to_t_start,
        info_jam_ft[idx_mon_oi_mod[2]]['free_parameter_values'][idx_pop_fp][idx_t_start] - lat_playback_to_t_start]
    # plot traces
    for i, trc in enumerate(traces):
        ax_trc.append(plt.subplot2grid((n_rows, n_cols), (i_rows_trc[i], i_cols_trc[i]),
                                       colspan=n_cols_trc, rowspan=n_rows_trc))
        # add artificial spikes by increasing the potential value at spiketimes by a predefined amount
        if artificial_spike_heights:
            for spk in spiketimes_trc[i]:
                samp_spike = int(round(spk / (info_jam_ft[0]['dt'] / b2.ms)))
                trc[samp_spike] = trc[samp_spike] + artificial_spike_heights
        # plot marker at the time the first spike would occur if there was no jamming inhibition
        ax_trc[i].axvline(t_first_spike_no_jam - offset_ms[i], color=color_map_rec[3], linewidth=2,
                          label='unperturbed model burst', linestyle=(0, (1, 1)))  # (0, (1, 1)) == densely dotted
        # plot trace
        dur_sample = 1 / samp_freqs[i]
        dur_trace = dur_sample * trc.size
        x_values = np.linspace(0, dur_trace, trc.size)
        ax_trc[i].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
        ax_trc[i].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')
        ax_trc[i].plot(x_values - offset_ms[i], trc, color=color_map_mod[3], linewidth=linewidth,
                       label='perturbed model trace')
        # format axis
        ax_trc[i].set_xlim(x_lim_traces)
        ax_trc[i].set_ylabel('$\mathrm{V_m}$ [mV]')
        if y_lim_traces[i]:
            ax_trc[i].set_ylim(y_lim_traces[i])
        if i < len(traces) - 1:
            ax_trc[i].set_xticklabels([])
        else:
            ax_trc[i].set_xlabel('Time from playback onset [ms]')
    ax_trc[0].legend()
    # endregion

    # region plot time of first spike of all trials per neuron ################################################
    ax_brst.append(plt.subplot2grid((n_rows, n_cols), (i_rows_brst[0], i_cols_brst[0]),
                                    colspan=n_cols_brst, rowspan=n_rows_brst))
    ax_brst[0].axvline(0, color=[0, 0, 0], linewidth=1, linestyle='--')
    t_first_spike_nrn = []
    t_first_spike_avg = []
    t_first_spike_std = []
    spiketimes_rec_all = [spiketimes_spiking, spiketimes_gabazine]
    for n, spikes_nrn in enumerate(spiketimes_rec_all[0]):
        t_first_spike_nrn.append([])
        for t, spikes_trial in enumerate(spikes_nrn):
            # NOTE: hardcoded criterion to exclude post-call spikes (100ms after call onset)
            if spikes_trial and spikes_trial[0] < rec_offset_ms_spiking[0][n] + 100:
                t_first_spike_nrn[n].append(spikes_trial[0])
        t_first_spike_avg.append(np.mean(t_first_spike_nrn[n]))
        t_first_spike_std.append(np.std(t_first_spike_nrn[n]))
    idx_nrn_sort_by_tfs = np.argsort(t_first_spike_avg)
    # plot first spike of each trial in the same row for all trials of one neuron
    for n, nrn in enumerate(idx_nrn_sort_by_tfs):
        for first_spike_trial in t_first_spike_nrn[nrn]:
            ax_brst[0].plot(
                (first_spike_trial - rec_offset_ms_spiking[0][n], first_spike_trial - rec_offset_ms_spiking[0][n]),
                (n + .5, n + 1.5), linewidth=1, color=color_map_rec[3])
    for n, nrn in enumerate(idx_nrn_sort_by_tfs):
        # plot the standard deviation
        ax_brst[0].plot((t_first_spike_avg[nrn] - t_first_spike_std[nrn] - rec_offset_ms_spiking[0][n],
                         t_first_spike_avg[nrn] + t_first_spike_std[nrn] - rec_offset_ms_spiking[0][n]), (n + 1, n + 1),
                        '-k', linewidth=1)
        # plot the average
        ax_brst[0].plot(t_first_spike_avg[nrn] - rec_offset_ms_spiking[0][n], n + 1, 'o', markeredgecolor='k',
                        markerfacecolor='w', markersize=4, markeredgewidth=1)
        print('mean, std of neuron ' + str(n) + ': ' + str(t_first_spike_avg[nrn]) + ', ' + str(t_first_spike_std[nrn]))
    # add marker for estimated time window of burst onsets susceptible to inhibitory suppression
    min_std_left = np.min([t_first_spike_avg[v] - t_first_spike_std[v] for v in range(len(t_first_spike_avg))])
    burst_suppression_win = (min_std_left - rec_offset_ms_spiking[0][0], -10)
    ax_brst[0].plot(burst_suppression_win, [len(idx_nrn_sort_by_tfs) - .8] * 2, '-', linewidth=3, color=[.9, .3, 0])
    ax_brst[0].text(np.mean(burst_suppression_win), len(idx_nrn_sort_by_tfs) - .4, 'est. window of\nsusceptibility',
                    va='bottom', ha='center', fontsize=FONTSIZE_XS, color=[.9, .3, 0])
    ax_brst[0].set_xlim(xlims_burst)
    ax_brst[0].set_ylim((0.5, len(idx_nrn_sort_by_tfs) + 0.5))
    ax_brst[0].set_xlabel('Time of burst onsets from call onset [ms]')
    ax_brst[0].set_ylabel('Observed premotor neuron\n(data from Benichov \& Vallentin, 2020)')
    # endregion

    # region plot overlap of time window after playback in which inhibition can suppress PM bursts and estimated time
    # window before call production in which PM suppression could cancel a call.
    # calculate time window for each potential call onset (1ms steps) in which bursts would be succeptible to
    # suppression (i.e. the suppression window estimated from burst onsets of recorded PMs)
    t_call_onset_sim_range = (-60, 160)
    x_lim_ovrlp = t_call_onset_sim_range
    t_call_onset_sim = np.array(range(t_call_onset_sim_range[0], t_call_onset_sim_range[1] + 1))
    burst_suppression_win_len = burst_suppression_win[1] - burst_suppression_win[0]
    burst_windows_ms_lr = [t_call_onset_sim + burst_suppression_win[0], t_call_onset_sim + burst_suppression_win[1]]
    t_overlap_ms = np.zeros(len(t_call_onset_sim))
    ratio_overlap_with_suppr_win = np.zeros(len(t_call_onset_sim))
    for i_call, t_call in enumerate(t_call_onset_sim):
        burst_win_l = burst_windows_ms_lr[0][i_call]
        burst_win_r = burst_windows_ms_lr[1][i_call]
        b_burst_win_r_in_suppr_win = suppr_win[0] < burst_win_r < suppr_win[1]
        b_burst_win_l_in_suppr_win = suppr_win[0] < burst_win_l < suppr_win[1]
        if b_burst_win_r_in_suppr_win and not b_burst_win_l_in_suppr_win:  # late bursts suppressed
            t_overlap_ms[i_call] = burst_win_r - suppr_win[0]
        elif b_burst_win_r_in_suppr_win and b_burst_win_r_in_suppr_win:  # all bursts suppressed
            t_overlap_ms[i_call] = burst_win_r - burst_win_l
        elif b_burst_win_l_in_suppr_win and not b_burst_win_r_in_suppr_win:  # early bursts suppressed
            t_overlap_ms[i_call] = suppr_win[1] - burst_win_l
        elif burst_win_l < suppr_win[0] and burst_win_r > suppr_win[1]:  # middle bursts suppressed
            t_overlap_ms[i_call] = suppr_win[1] - suppr_win[0]
        ratio_overlap_with_suppr_win[i_call] = t_overlap_ms[i_call] / burst_suppression_win_len
    # plot examples of call onsets relative to playback and visualize the overlap of the two windows
    ax_ovrlp.append(plt.subplot2grid((n_rows, n_cols), (i_rows_ovrlp[0], i_cols_ovrlp[0]),
                                     colspan=n_cols_ovrlp, rowspan=n_rows_ovrlp[0]))
    ax_ovrlp[0].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
    ax_ovrlp[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')
    ax_ovrlp[0].axvspan(suppr_win[0], suppr_win[1], facecolor=[.8, .8, .8], alpha=0.5, zorder=1)
    ax_ovrlp[0].set_xticklabels([])
    # index to t_call_onset_sim
    i_example_onsets = list(range(25 - t_call_onset_sim_range[0], 125 - t_call_onset_sim_range[0] + 1, 25))
    for i_ex, idx_onset in enumerate(i_example_onsets):
        # plot window of susceptibility
        ax_ovrlp[0].plot((burst_windows_ms_lr[0][idx_onset], burst_windows_ms_lr[1][idx_onset]),
                         (t_call_onset_sim[idx_onset], t_call_onset_sim[idx_onset]), '-', linewidth=3,
                         color=[.9, .3, 0],
                         zorder=0)
        # plot call onsets
        ax_ovrlp[0].plot((t_call_onset_sim[idx_onset], t_call_onset_sim[idx_onset]),
                         (t_call_onset_sim[idx_onset] - 10, t_call_onset_sim[idx_onset] + 10), '--k', linewidth=1,
                         zorder=2)
    ax_ovrlp[0].set_xlim(x_lim_ovrlp)
    # ax_ovrlp[0].set_xlabel('Time from playback onset [ms]')
    ax_ovrlp[0].set_ylabel('Time of call\nonset [ms]')
    ax_ovrlp[0].set_yticks([t_call_onset_sim[i] for i in i_example_onsets])
    # plot percentage of estimated pre-call suppression window with
    # (estimated percentage of PM bursts, that would be suppressed for each call time re. to playback).
    ax_ovrlp.append(plt.subplot2grid((n_rows, n_cols), (i_rows_ovrlp[1], i_cols_ovrlp[1]),
                                     colspan=n_cols_ovrlp, rowspan=n_rows_ovrlp[1]))
    ax_ovrlp[1].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
    ax_ovrlp[1].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')
    for i_ex, idx_onset in enumerate(i_example_onsets):
        ax_ovrlp[1].axvline(t_call_onset_sim[idx_onset], color=[0, 0, 0], linewidth=1, linestyle='--')
    ax_ovrlp[1].plot(t_call_onset_sim, ratio_overlap_with_suppr_win * 100, color=[.5, .5, .5])
    ax_ovrlp[1].set_xlim(x_lim_ovrlp)
    ax_ovrlp[1].set_ylim((-10, np.max(ratio_overlap_with_suppr_win * 100) + 10))
    ax_ovrlp[1].set_xlabel('Time from playback onset [ms]')
    ax_ovrlp[1].set_ylabel('% suppr.')
    # endregion

    # region histograms of simulated and recorded call onset times ########################################
    # create pseudorandomly, uniformly distributed call onsets w/o and w/ suppression
    prng = np.random.RandomState(19)
    call_onsets_rand = prng.uniform(t_call_onset_sim_range[0], t_call_onset_sim_range[1], 150)
    idx_closest_sim_onset = [np.where(call_onsets_rand[i] < (t_call_onset_sim + 0.5))[0][0]
                             for i in range(len(call_onsets_rand))]
    call_onsets_suppr = []
    for i, c in enumerate(call_onsets_rand):
        if prng.uniform() < (1 - ratio_overlap_with_suppr_win[idx_closest_sim_onset[i]] * 1.5):
            call_onsets_suppr.append(c)

    # shift end of distribution of recorded calls to beginning (before 0)
    i_calls_to_shift = np.where(call_onset_times['Control_1hz_call_onsets'] > 1000 + t_call_onset_sim_range[0])[0]
    call_onsets_control = call_onset_times['Control_1hz_call_onsets'].astype(int)
    call_onsets_control_shifted = call_onsets_control.copy()
    call_onsets_control_shifted[i_calls_to_shift] = call_onsets_control[i_calls_to_shift] - 1000
    i_calls_to_shift = np.where(call_onset_times['Gabazine_1hz_call_onsets'] > 1000 + t_call_onset_sim_range[0])[0]
    call_onsets_gabazine = call_onset_times['Gabazine_1hz_call_onsets'].astype(int)
    call_onsets_gabazine_shifted = call_onsets_gabazine.copy()
    call_onsets_gabazine_shifted[i_calls_to_shift] = call_onsets_gabazine[i_calls_to_shift] - 1000
    call_onsets_shifted = [call_onsets_suppr, call_onsets_control_shifted, call_onsets_gabazine_shifted]
    clr = [color_map_mod[3], color_map_rec[3], color_map_rec[4]]
    bins_n = []
    for p in range(len(i_rows_hist)):
        ax_hist.append(plt.subplot2grid((n_rows, n_cols), (i_rows_hist[p], i_cols_hist[p]),
                                        colspan=n_cols_hist, rowspan=n_rows_hist))
        ax_hist[p].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
        if p == 0:
            n, _, _ = ax_hist[p].hist(call_onsets_rand, color=[.8, .8, .8], histtype='stepfilled',
                                      bins=np.arange(t_call_onset_sim_range[0], t_call_onset_sim_range[1] + 20, 20))
        else:
            n, _, _ = ax_hist[p].hist(call_onsets_shifted[p - 1], color=clr[p - 1], histtype='stepfilled',
                                      bins=np.arange(t_call_onset_sim_range[0], t_call_onset_sim_range[1] + 20, 20))
        bins_n.append(n)
        ax_hist[p].set_xlim(x_lim_ovrlp)
        if p < 2:
            ax_hist[p].set_ylabel('N simulated\ncall onsets')
        else:
            ax_hist[p].set_ylabel('N observed\ncall onsets')
        ax_hist[p].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
    max_bin = np.max([np.max(bins) for bins in bins_n[0:3]])
    [ax_hist[i].set_ylim((0, max_bin)) for i in range(len(ax_hist))]
    for i in range(len(ax_hist) - 1):
        ax_hist[i].set_xlabel(' ')
        ax_hist[i].tick_params(labelbottom=False)
    ax_hist[-1].set_xlabel('Time from playback onset [ms]')
    ax_hist[0].text(66, max_bin - 13, 'simulation\n(no suppression)', ha='center', size=FONTSIZE_S, color=[.7, .7, .7])
    ax_hist[1].text(42, max_bin - 7, 'simulation', ha='center', size=FONTSIZE_S, color=clr[0])
    ax_hist[2].text(41, max_bin - 7, 'control', ha='center', size=FONTSIZE_S, color=clr[1])
    ax_hist[3].text(40, max_bin - 7, 'gabazine', ha='center', size=FONTSIZE_S, color=clr[2])
    # add red borders to observed histograms
    width_zoom = sum([abs(t) for t in t_call_onset_sim_range])
    zoom_area = matplotlib.patches.Rectangle((t_call_onset_sim_range[0], 0), width_zoom, max_bin, zorder=5,
                                             linestyle='-', linewidth=1.5, color='r', fill=False, clip_on=False)
    ax_hist[2].add_patch(zoom_area)
    zoom_area = matplotlib.patches.Rectangle((t_call_onset_sim_range[0], 0), width_zoom, max_bin, zorder=5,
                                             linestyle='-', linewidth=1.5, color='r', fill=False, clip_on=False)
    ax_hist[3].add_patch(zoom_area)
    # endregion

    # region full histograms of recorded call onset times
    ax_hist_all.append(plt.subplot2grid((n_rows, n_cols), (i_rows_hist_all[0], i_cols_hist_all[0]),
                                        colspan=n_cols_hist_all, rowspan=n_rows_hist_all))
    ax_hist_all[0].axvline(0, color=[0, 0, 0], linewidth=1, linestyle=':')
    ax_hist_all[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':')  # playback offset
    hist_labels = ['', 'control', 'gabazine']
    for p in [1, 2]:
        n, _, _ = ax_hist_all[0].hist(call_onsets_shifted[p], histtype='stepfilled', color=clr[p] + [0.6],
                                      bins=np.arange(t_call_onset_sim_range[0],
                                                     1000 + t_call_onset_sim_range[0] + 20, 20), label=hist_labels[p])
    for p in [1, 2]:
        n, _, _ = ax_hist_all[0].hist(call_onsets_shifted[p], histtype='step', edgecolor='k', linewidth=.2,
                                      bins=np.arange(t_call_onset_sim_range[0],
                                                     1000 + t_call_onset_sim_range[0] + 20, 20))
    bins_n.append(n)
    ax_hist_all[0].set_xlim((t_call_onset_sim_range[0], 1000 + t_call_onset_sim_range[0]))
    ax_hist_all[0].set_ylim((0, 150))
    ax_hist_all[0].set_ylabel('N observed\ncall onsets')
    ax_hist_all[-1].set_xlabel('Time from playback onset [ms]')
    ax_hist_all[0].legend(framealpha=0.9)
    zoom_area = matplotlib.patches.Rectangle((t_call_onset_sim_range[0], 0), width_zoom, max_bin, zorder=5,
                                             linestyle='-', linewidth=1.5, color='r', fill=False, clip_on=False)
    ax_hist_all[0].add_patch(zoom_area)
    # endregion

    # add sub-figure labels
    ax_crc[0].annotate('A', xy=(-1.07, 1.04), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_lat[0].annotate('B', xy=(-1.1 / n_cols_lat, 1.2), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_trc[0].annotate('C', xy=(-1.1 / n_cols_trc, 1.08), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_brst[0].annotate('D', xy=(-1 / n_cols_brst, 1.04), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_hist_all[0].annotate('G', xy=(-1 / n_cols_hist_all, 1.055), xycoords='axes fraction', fontweight='bold',
                            size=FONTSIZE_XL)
    ax_ovrlp[0].annotate('E', xy=(-1.1 / n_cols_ovrlp, 1.09), xycoords='axes fraction', fontweight='bold',
                         size=FONTSIZE_XL)
    ax_hist[0].annotate('F', xy=(-1.1 / n_cols_hist, 1.1), xycoords='axes fraction', fontweight='bold',
                        size=FONTSIZE_XL)
    ax_tpr[0].annotate('H', xy=(-1.25 / n_cols_tpr, 1.055), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    return fig, ax_brst + ax_ovrlp + ax_hist + ax_hist_all + ax_tpr + ax_lat + ax_trc + [h_cb.ax]


def fig_s_sensitivity(config_name_sens_lo, run_id_sens, idx_marked_sim=None, b_mark_weight_sims=False):
    color_map_mod = p_plot.get_color_list(7, color_map_in=[C_COLORS[c] for c in COL_SEQ])

    if run_id_sens:
        b_run_id_in_filename = True
    else:
        b_run_id_in_filename = False
    fig, ax = routine.sensitivity_analysis(config_name_sens_lo, run_id_sens, b_show_figures=False,
                                           b_run_id_in_filename=b_run_id_in_filename)
    if idx_marked_sim is not None:
        for i, a in enumerate(ax):
            # color mark relevant simulations
            if i == idx_marked_sim:
                for side in ['top', 'bottom', 'left', 'right']:
                    a.spines[side].set_color(color_map_mod[3])
                    a.spines[side].set_linewidth(3)
                    a.spines[side].set_visible(True)
    if b_mark_weight_sims:
        for i, a in enumerate(ax):
            if i in [51, 62, 73, 84, 95, 106, 117, 128, 139]:
                for side in ['top', 'bottom', 'left', 'right']:
                    a.spines[side].set_color(color_map_mod[4])
                    a.spines[side].set_linewidth(3)
                    a.spines[side].set_visible(True)
            elif i == 34:
                for side in ['top', 'bottom', 'left', 'right']:
                    a.spines[side].set_color(color_map_mod[2])
                    a.spines[side].set_linewidth(3)
                    a.spines[side].set_visible(True)

    return fig, ax


def fig_s_current(states_curr, info_curr, config_curr):
    axes = []
    fig = plt.figure(figsize=(10, 6))
    fig.tight_layout()
    plt.rcParams.update({'font.size': 9})
    n_pops = len(info_curr[0]['population_sizes'])
    n_gridcols = np.round(n_pops / 2).astype(int)
    n_gridrows = 2
    matplotlib.gridspec.GridSpec(n_gridrows, n_gridcols)
    color_map_mod = p_plot.get_color_list(4, color_map_in=[C_COLORS[c] for c in COL_SEQ], i_colors=[0, 0, 6, 6])

    # loop through populations
    marker_linestyle = ['--', '--', ':', ':']
    for pop in range(n_pops):
        axes.append(plt.subplot2grid((n_gridrows, n_gridcols), (pop % 2, np.floor(pop / 2).astype(int)),
                                     colspan=1, rowspan=1))
        # plot a vertical line to mark call/playback onset
        axes[pop].axvline(0, color='k', linewidth=1, linestyle=marker_linestyle[pop], zorder=1)
        # plot input current
        axes[pop].plot(states_curr[0].t / b2.ms - config_curr['misc']['playback_start'],
                       states_curr[0].Ie[pop, :] / b2.pA, color=color_map_mod[pop], linewidth=1, label='')
        # other settings
        axes[pop].set_xlabel(' ')
        axes[pop].set_ylabel('$\mathbf{I_e}$ [pA]')
        axes[pop].set_ylim((70, 300))
        axes[pop].set_xlim((-100, 100))

    axes[1].set_xlabel('Time from call onset [ms]')
    axes[3].set_xlabel('Time from playback onset [ms]')

    # share x axis (time) among all  plots
    [axes[n].get_shared_x_axes().join(axes[n], axes[n + 1]) for n in range(n_pops - 1)]
    [axes[n].get_shared_y_axes().join(axes[n], axes[n + 2]) for n in (0, 1)]

    # add sub-figure labels
    axes[0].annotate('A', xy=(-0.16, 1.0), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    axes[2].annotate('B', xy=(-0.16, 1.0), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    return fig, axes


def fig_s_psp(states_psp, config_psp, info_psp):
    fig_psp, ax_psp = p_plot.plot_psp_psc(states_psp, config_psp, info_psp, i_colors=[3, 0, 0])

    # add sub-figure labels
    ax_psp[0][1].annotate('A', xy=(-0.16, 1.0), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psp[0][0].annotate('B', xy=(-0.16, 1.0), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.25, hspace=0.3)

    return fig_psp, ax_psp[0]


def fig_s_prespike_ramp(states_ag0, spikes_ag0, info_ag0,
                        states_ag1, spikes_ag1, info_ag1, states_ag2, spikes_ag2, info_ag2,
                        states_ag1_lowin, spikes_ag1_lowin, info_ag1_lowin,
                        states_ag2_lowin, spikes_ag2_lowin, info_ag2_lowin,
                        traces_spiking, spiketimes_spiking, traces_gabazine, spiketimes_gabazine):
    # figure parameters
    n_rows = 15
    n_cols = 8
    i_rows_psa = [0, 2, 5, 7, 10, 12]
    i_cols_psa = [4, 4, 4, 4, 4, 4]
    n_rows_psa = 2
    n_cols_psa = 2
    i_rows_dot = [0, 5, 10]
    i_cols_dot = [7, 7, 7]
    n_rows_dot = 4
    n_cols_dot = 1
    i_rows_crc_big = [0, 5, 10]
    i_cols_crc_big = [0, 0, 0]
    n_rows_crc_big = 4
    n_cols_crc_big = 3

    # file, population & trace indices
    samp_freq_rec = 40
    letter_pop_mod = ['V', '$\mathbf{I_v}$', 'P', 'P', 'P', 'T']
    dur_prespike = 50
    t1_prespike = 0  # this minus dur_prespike in the plot; same for t2
    t2_prespike = dur_prespike - 10
    color_map_rec = p_plot.get_color_list(7, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])
    color_map_mod = p_plot.get_color_list(7, color_map_in=[C_COLORS[c] for c in COL_SEQ])

    # set up figure
    fig = plt.figure(figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_crc_big = []
    ax_psa = []
    ax_dot = []

    # region plot big circuit diagrams on top ###################################################################
    # alternative diagram with no-inhibition model PMs receiving motor-related input as well
    x_circuit = [[1, 1, 1, 2], [1, 1, 1, 2, 2], [1, 1, 1, 2]]
    y_circuit = [[1.5, 0.5, 2.5, .5], [1.5, 0.5, 2.5, 1.5, .5], [1.5, 0.5, 2.5, .5]]
    conn_circ = [[(0, 1), (0, 3), (2, 0)], [(0, 1), (0, 4), (3, 4), (2, 0)], [(0, 1), (0, 3), (1, 3), (2, 0)]]
    conn_type = [[1.5, 0.5, 1.5], [1.5, 1.5, -1.5, 1.5], [1.5, 1.5, -1.5, 1.5]]
    conn_colors = [['k', 'k', 'k'], ['k', 'k', [1, 0, 0.4], 'k'], ['k', 'k', [1, 0, 0.4], 'k']]
    color_circ = [[color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[3]],
                  [color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[5], color_map_mod[3]],
                  [color_map_mod[0], color_map_mod[1], [0, 0, 0], color_map_mod[3]]]
    letters = [[letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[5], letter_pop_mod[3]],
               [letter_pop_mod[0], letter_pop_mod[1], None, letter_pop_mod[3]]]
    ramp_current = np.concatenate((np.zeros(5), [math.pow(-x / 70, 2) * 0.1 for x in range(0, 70 + 1)],
                                   np.linspace(0.1, 0, 10), np.zeros(20)))
    currents = [[None, None, ramp_current, None], [None, None, ramp_current, None, None],
                [None, None, ramp_current, None]]
    # set axis limits and plot
    x_lim_crc = (min([min(v) for v in x_circuit]) - 0.7, max([max(v) for v in x_circuit]) + 0.3)
    y_lim_crc = (min([min(v) for v in y_circuit]) - 0.3, max([max(v) for v in y_circuit]) + 0.7)
    text_mod = ['No inhibition', 'Tonic inhibition', 'Feed-forward\ninhibition']
    for i in range(len(x_circuit)):
        ax_crc_big.append(plt.subplot2grid((n_rows, n_cols), (i_rows_crc_big[i], i_cols_crc_big[i]),
                                           colspan=n_cols_crc_big, rowspan=n_rows_crc_big))
        circuit_diagram(ax_crc_big[i], x_circuit[i], y_circuit[i], color_circ[i], conn_circ[i], conn_type[i],
                        letters=letters[i], currents=currents[i], x_lim=x_lim_crc, y_lim=y_lim_crc,
                        letter_size=FONTSIZE_XL, connection_colors=conn_colors[i])
        ax_crc_big[i].text(1.5, 3.555, text_mod[i], va='top', ha='center', fontsize=FONTSIZE_M, fontweight='bold')
    # endregion

    # region plot prespike averages of recorded premotor neurons and model neurons (100 run random current avg.)
    traces_rec_all = [traces_spiking, traces_gabazine]
    spiketimes_rec_all = [spiketimes_spiking, spiketimes_gabazine]
    states_mod = [states_ag0, states_ag0, states_ag2, states_ag2_lowin, states_ag1, states_ag1_lowin]
    spikes_mod = [spikes_ag0, spikes_ag0, spikes_ag2, spikes_ag2_lowin, spikes_ag1, spikes_ag1_lowin]
    info_mod = [info_ag0, info_ag0, info_ag2, info_ag2_lowin, info_ag1, info_ag1_lowin]
    i_traces_rec_psa = [0, 1, 0, 1, 0, 1]
    i_spiketimes_rec_psa = [0, 1, 0, 1, 0, 1]
    i_colors_psa = [3, 4, 3, 4, 3, 4]
    i_pops_ag = [3, 3, 3, 3, 2, 2]
    v_t1 = []  # voltages of prespike average at timepoint 1 (see above t1_prespike)
    v_t2 = []
    for p in range(len(i_cols_psa)):
        ax_psa.append(plt.subplot2grid((n_rows, n_cols), (i_rows_psa[p], i_cols_psa[p]),
                                       colspan=n_cols_psa, rowspan=n_rows_psa))
        v_t1.append([])
        v_t2.append([])
        # plot marker for t2, as well as labels for the timepoints at which Vm difference is measured for ax_dot (t1,t2)
        ax_psa[p].axvline(t2_prespike - dur_prespike, color=[.8, .8, .8])
        if p % 2 == 0:
            ax_psa[p].text(t1_prespike - dur_prespike, 3, 't1', va='bottom', ha='center', fontsize=FONTSIZE_S)
            ax_psa[p].text(t2_prespike - dur_prespike, 3, 't2', va='bottom', ha='center', fontsize=FONTSIZE_S)
        # get and plot recorded traces preceding and aligned to first spike
        smoothed_prespike_averages = []
        n_traces_in_prespike_avg = []
        for i in range(len(traces_rec_all[i_traces_rec_psa[p]])):
            smoothed_timeseries_average, smoothed_timeseries_all, n_traces_included = \
                p_analysis.prespike_trace_average(traces_rec_all[i_traces_rec_psa[p]][i],
                                                  spiketimes_rec_all[i_spiketimes_rec_psa[p]][i], duration=dur_prespike)
            v_t1[p].append(smoothed_timeseries_average[int(round(t1_prespike * samp_freq_rec))])
            v_t2[p].append(smoothed_timeseries_average[int(round(t2_prespike * samp_freq_rec))])
            smoothed_prespike_averages.append(smoothed_timeseries_average)
            n_traces_in_prespike_avg.append(n_traces_included)
            dur_sample = 1 / samp_freq_rec
            dur_trace = dur_sample * smoothed_timeseries_average.size
            x_values = np.linspace(0, dur_trace, smoothed_timeseries_average.size)
            h_rec, = ax_psa[p].plot(x_values - dur_prespike, smoothed_timeseries_average,
                                    color=color_map_rec[i_colors_psa[p]], linewidth=1)
        # get and plot prespike average of aggregate model data
        spiketimes_ms_ag = [p_util.get_spiketimes_from_monitor(spikes_mod[p][m], info_mod[p][m], i_pops_ag[p])
                            for m in range(len(spikes_mod[p]))]
        traces_ag, samp_freq_ag = p_util.get_traces_from_monitors(states_mod[p])
        # get prespike trace average
        smoothed_timeseries_ag_avg, _, n_traces_included = p_analysis.prespike_trace_average(
            [traces_ag[i][0] / b2.mV for i in range(len(traces_ag))],
            [spiketimes_ms_ag[i][0] for i in range(len(spiketimes_ms_ag))], duration=dur_prespike,
            b_spiketime_is_onset=True, sampling_frequency_khz=samp_freq_ag, b_smooth=False)
        v_t1[p].append(smoothed_timeseries_ag_avg[int(round(t1_prespike * samp_freq_ag))])
        v_t2[p].append(smoothed_timeseries_ag_avg[int(round(t2_prespike * samp_freq_ag))])
        smoothed_prespike_averages.append(smoothed_timeseries_ag_avg)
        n_traces_in_prespike_avg.append(n_traces_included)
        # plot aggregate model prespike average
        dur_sample = 1 / samp_freq_ag
        dur_trace = dur_sample * smoothed_timeseries_ag_avg.size
        x_values = np.linspace(0, dur_trace, smoothed_timeseries_ag_avg.size)
        h_mod, = ax_psa[p].plot(x_values - dur_prespike, smoothed_timeseries_ag_avg,
                                color=color_map_mod[i_colors_psa[p]], linewidth=2)
        y_lim_psa = (-55, 0)
        ax_psa[p].legend(handles=(h_rec, h_mod), labels=('obs', 'mod'), loc='lower left', handlelength=1, ncol=2)
        ax_psa[p].set_xlim((-dur_prespike, 0))
        ax_psa[p].set_ylim(y_lim_psa)
        ax_psa[p].set_xlabel('Time from first spike [ms]')
        ax_psa[p].set_ylabel('$\mathrm{V_m}$ [mV]')
        # for every second plot remove y axis labels
        if p % 2 == 0:
            ax_psa[p].set_xlabel('')
            ax_psa[p].set_xticklabels('')
    # endregion

    # region dot plots of difference in potential at two time points prior to first spike (i.e. ramp "magnitude")
    for g in range(len(i_cols_dot)):
        ax_dot.append(plt.subplot2grid((n_rows, n_cols), (i_rows_dot[g], i_cols_dot[g]),
                                       colspan=n_cols_dot, rowspan=n_rows_dot))
        v_diff_control = [v_t2[g * 2][v] - v_t1[g * 2][v] for v in range(len(v_t1[g * 2]))]
        v_diff_gabazine = [v_t2[g * 2 + 1][v] - v_t1[g * 2 + 1][v] for v in range(len(v_t1[g * 2 + 1]))]
        clr = [[color_map_rec[i_colors_psa[g * 2]]] * (len(v_diff_control) - 1) + [color_map_mod[i_colors_psa[g * 2]]]]\
            + [[color_map_rec[i_colors_psa[g * 2 + 1]]] * (len(v_diff_gabazine) - 1) \
            + [color_map_mod[i_colors_psa[g * 2 + 1]]]]
        mrkr = [[5] * (len(v_diff_control) - 1) + [8], [5] * (len(v_diff_gabazine) - 1) + [8]]
        [print('v_diff_control: ' + str(v)) for v in v_diff_control]
        [print('v_diff_gabazine: ' + str(v)) for v in v_diff_gabazine]
        [print('clr: ' + str(c)) for c in clr]
        ax_dot[g].axhline(0, color=[.8, .8, .8], linewidth=1)
        p_plot.dot_plot([v_diff_control, v_diff_gabazine], ['control', 'gabazine'], '', ' ', value_colors=clr,
                        b_mean=False, b_jitter=True, jitter_thresh=0.7, jitter_offset=0.25, marker_size=mrkr,
                        h_ax=ax_dot[g])
        ax_dot[g].set_ylim([-2, 38])
        plt.xticks(rotation=40)
        ax_dot[g].set_ylabel('$\mathrm{V_m}$ difference (t2 - t1) [mV]')
    # endregion

    # add sub-figure labels
    ax_crc_big[0].annotate('A', xy=(-0.01, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc_big[1].annotate('D', xy=(-0.01, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_crc_big[2].annotate('G', xy=(-0.01, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psa[0].annotate('B', xy=(-0.44, 1.13), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psa[2].annotate('E', xy=(-0.44, 1.13), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_psa[4].annotate('H', xy=(-0.44, 1.13), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_dot[0].annotate('C', xy=(-0.9, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_dot[1].annotate('F', xy=(-0.9, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)
    ax_dot[2].annotate('I', xy=(-0.9, 1.06), xycoords='axes fraction', fontweight='bold', size=FONTSIZE_XL)

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    return fig, ax_psa + ax_dot


def fig_s_depolarized(traces_playback_aligned):
    # figure parameters
    n_rows = 15
    n_cols = 19
    i_rows_sub = [0]  # playback-aligned subthreshold potentials of bursting PM
    i_cols_sub = [0]
    n_rows_sub = 2
    n_cols_sub = 4
    x_lim_agg = [-100, 165]
    color_map_rec = p_plot.get_color_list(8, color_map_in=[C_COLORS_LIGHT[c] for c in COL_SEQ])

    # file, population & trace indices
    call_onset_ms_playback_aligned = 200

    # set up figure
    fig = plt.figure(figsize=(n_cols * 0.888, n_rows * 0.75))
    matplotlib.gridspec.GridSpec(n_rows, n_cols)
    ax_sub = []

    # region playback-aligned observed PM
    ax_sub.append(plt.subplot2grid((n_rows, n_cols), (i_rows_sub[0], i_cols_sub[0]),
                                   colspan=n_cols_sub, rowspan=n_rows_sub))
    samp_freq_khz = 40
    downsampling_factor = 20
    trace_playback_aligned_trim = p_util.trim_traces_to_time_span([call_onset_ms_playback_aligned + x_lim_agg[0],
                                                                   call_onset_ms_playback_aligned + x_lim_agg[1]],
                                                                  traces_playback_aligned[0], samp_freq_khz)
    trace_playback_aligned_trim_down = p_util.downsample_trace(trace_playback_aligned_trim, downsampling_factor)
    # DE-MEAN
    for i in range(len(trace_playback_aligned_trim_down)):
        trace_playback_aligned_trim_down[i] = trace_playback_aligned_trim_down[i] - np.mean(
            trace_playback_aligned_trim_down[i])
    p_plot.plot_traces_rec(trace_playback_aligned_trim_down, sampling_frequency_khz=samp_freq_khz / downsampling_factor,
                           b_plot_raw_traces=False, b_plot_smoothed=True, b_plot_average=True, b_std_interval=False,
                           b_offset_by_average=False, b_n_in_legend=False, t_const_event_marker=0, marker_linestyle=':',
                           trace_color_smooth=[.85, .85, .85], average_color=color_map_rec[3], t_offset_ms=100,
                           b_average_mean_2std=True, t_baseline_ms=[0, 100], h_ax=ax_sub[0])
    ax_sub[0].axvline(110, color=[0, 0, 0], linewidth=1, linestyle=':', zorder=2.5)  # playback offset
    h_l = ax_sub[0].legend(loc='upper left')
    h_l.get_texts()[0].set_text('observed')
    h_l.get_texts()[1].set_text('average')
    ax_sub[0].set_xlabel('Time from playback onset [ms]')
    ax_sub[0].set_ylabel('$\mathrm{V_m}$ [mV]')
    ax_sub[0].set_xlim(x_lim_agg)
    ax_sub[0].set_ylim((-5, 5))
    # endregion

    # adjust subplot margins
    plt.subplots_adjust(wspace=0.4, hspace=0.8)

    return fig, ax_sub


def circuit_diagram(h_ax, x, y, colors, connectivity, connection_type, connection_colors=None, letters=None,
                    currents=None, letter_size=12, x_lim=None, y_lim=None, region_rectangle_xywh=None,
                    region_label=None, linestyle=None, linewidth_circle=2):
    """Draw a circuit diagram in axis h_ax, consisting of populations represented as colored circles, whose positions
    and colors are given by x, y and colors (index to those is population id). Connections between populations
    are drawn according to connectivity: one tuple of pre and post population id per connection. Arrowheads for each
    (triangle/circle) are given by connection_type (-1/1 corresponds to excitatory/inhibitory).

    :param h_ax: axis handle
    :param x: list of x-axis positions for the populations. one value per population.
    :param y: list of y-axis positions for the populations. one value per population.
    :param colors: list of color values for the population circles . one value (e.g. RGB tuple) per population.
        where the value is None, the corresponding array in currents will be plotted instead of a population circle.
    :param connectivity: list of 2-element tuples givin id of pre- and post-population (corresponding to x/y indices)
    :param connection_type: list of negative (inhibitory) or positive (excitatory) floats with same length as
        connectivity (n connections). Absolute size of the floats or ints determins line width of the connecting lines.
    :param connection_colors: list of color values for the connecting lines with same length as connectivity.
    :param letters: optional list of strings (letters/symbols) or Nones, one per population, to be printed in the circle
    :param letter_size: font size of the letters
    :param currents: optional list of arrays of input currents. list length should be number of populations. should be
        None except where a missing colors value (i.e. None) indicates that a current should be plotted instead of a pop
        circle. minimum should be 0 for proper scaling.
    :param x_lim: x-axis limits. if None (default), adjust automatically
    :param y_lim: y-axis limits. if None (default), adjust automatically
    :param region_rectangle_xywh: list of x, y, width, height values for a gray rectangle indicating a brain region.
        x and y are the positions of the bottom left population (circle) in the region. width/height is the number of
        populations in x/y direction included in the region, minus one (i.e. single population at 1, 1: [1, 1, 0, 0])
    :param region_label: string label for region, to be plotted below rectangle
    :param linestyle: list of strings of linestyles for the connecting lines (e.g. '--', ':'). default: '-'
    :param linewidth_circle: float linewidth used for all circles (neurons/populations). Default = 2. Current: - 0.5
    :return:
    """

    crc_radius = 0.25  # radius of circles representing neuron populations
    inh_conn_radius = 0.08  # radius of circles representing inhibitory connections
    arrow_headsize = 0.15  # size of triangles representing excitatory connections

    if connection_colors is None:
        connection_colors = ['k'] * len(connectivity)
    if linestyle is None:
        linestyle = ['-'] * len(connectivity)

    # plot rectangle indicating brain region
    if region_rectangle_xywh:
        rect_x = region_rectangle_xywh[0] - 0.2
        rect_y = region_rectangle_xywh[1] - 0.2
        rect_w = region_rectangle_xywh[2] + 0.4
        rect_h = region_rectangle_xywh[3] + 0.4
        if region_label:
            rect_y -= 0.3
            rect_h += 0.3
            h_ax.text(rect_x + rect_w / 2, rect_y + 0.1, region_label, fontweight='bold', fontsize=letter_size,
                      color=[.7, .7, .7], ha='center', va='top')
        box = matplotlib.patches.FancyBboxPatch((rect_x, rect_y), rect_w, rect_h, edgecolor=[.7, .7, .7],
                                                facecolor='w', linewidth=2)
        box.set_clip_on(False)
        h_ax.add_patch(box)

    # plot a line between circles for each connection & triangles/circles representing endpoints of exc/inh connections
    for i, c in enumerate(connectivity):
        h_ax.plot((x[c[0]], x[c[1]]), (y[c[0]], y[c[1]]), color=connection_colors[i], zorder=1,
                  linewidth=abs(connection_type[i]), linestyle=linestyle[i], clip_on=False)
        # plot arrows for excitatory synapses
        if connection_type[i] > 0:
            angle_rad = np.arctan2(y[c[0]] - y[c[1]], x[c[0]] - x[c[1]])  # pre_y - post_y, pre_x - post_x
            arrow_len = [crc_radius * np.cos(angle_rad), crc_radius * np.sin(angle_rad)]
            plt.arrow(x[c[1]], y[c[1]], arrow_len[0] * 1.1, arrow_len[1] * 1.1, facecolor=connection_colors[i],
                      edgecolor=connection_colors[i], head_width=arrow_headsize, head_length=arrow_headsize, zorder=1,
                      clip_on=False)
        # plot small circles for inhibitory synapses
        elif connection_type[i] < 0:
            angle_rad = np.arctan2(y[c[0]] - y[c[1]], x[c[0]] - x[c[1]])  # pre_y - post_y, pre_x - post_x
            crc_dist = [(crc_radius + inh_conn_radius + 0.035) * np.cos(angle_rad),
                        (crc_radius + inh_conn_radius + 0.035) * np.sin(angle_rad)]
            circle_in_syn = plt.Circle((x[c[1]] + crc_dist[0], y[c[1]] + crc_dist[1]), inh_conn_radius, facecolor='w',
                                       edgecolor=connection_colors[i], linewidth=abs(connection_type[i]), zorder=1)
            circle_in_syn.set_clip_on(False)
            h_ax.add_artist(circle_in_syn)

    # plot circles representing neuron populations or input currents
    for p in range(len(x)):
        if currents is not None and currents[p] is not None:
            circle_tmp = plt.Circle((x[p], y[p]), crc_radius * 5 / 4, edgecolor='w', facecolor='w', linewidth=2,
                                    zorder=2,
                                    clip_on=False)
            h_ax.add_artist(circle_tmp)  # white circle for background
            h_current, = plt.plot(np.linspace(x[p] - crc_radius * 5 / 6, x[p] + crc_radius * 5 / 6, len(currents[p])),
                                  np.array(currents[p]) / max(currents[p]) * crc_radius * 5 / 4 + y[
                                      p] - crc_radius * 2 / 4,
                                  '-', linewidth=1.5, zorder=3, clip_on=False, color=colors[p])
            h_current.set_solid_capstyle('round')
            circle_tmp = plt.Circle((x[p], y[p]), crc_radius * 7 / 5, linestyle='--', edgecolor=colors[p],
                                    facecolor='none', linewidth=linewidth_circle - 1, zorder=4, clip_on=False)
            h_ax.add_artist(circle_tmp)  # dashed, grey circle surrounding current
        else:
            circle_tmp = plt.Circle((x[p], y[p]), crc_radius, edgecolor=colors[p], facecolor='w',
                                    linewidth=linewidth_circle, zorder=2, clip_on=False)
            h_ax.add_artist(circle_tmp)
            if letters is not None and letters[p]:
                h_ax.text(x[p], y[p], letters[p], color=colors[p], weight='bold', size=letter_size,
                          ha='center', va='center')

    # set net axis aspect equal (i.e. resizing figure will not affect shape of axis) and other formatting
    h_ax.set_aspect('equal')
    h_ax.axis('off')
    if x_lim:
        h_ax.set_xlim(x_lim)
    else:
        h_ax.set_xlim((min(x) - 0.5, max(x) + 0.5))
    if y_lim:
        h_ax.set_ylim(y_lim)
    else:
        h_ax.set_ylim((min(y) - 0.5, max(y) + 0.5))


def load_model_data(model_name):
    """Get simulation results from .pkl file.

    :param model_name: model name (filename of output pickle file without ".pkl" ending)
    :type model_name: str
    :return: output variables (monitors, connectivity, config, info) as returned by p_io.load_monitors()
    """

    filename_pkl = p_io.BASE_PATH_OUTPUT + 'out' + os.path.sep + model_name + '.pkl'
    return p_io.load_monitors(filename_pkl)


def load_recording_data(path_to_traces, path_to_spiketimes):
    """Get voltage traces from all .mat files in path_to_traces.

    :param path_to_traces: path to directory containing .mat files of recorded traces
    :type path_to_traces: str or None
    :param path_to_spiketimes: path to directory containing .pkl files with spiketimes of recorded traces
    :type path_to_spiketimes: str or None
    :return:    - traces_rec: list of lists of recorded traces, one sub-list per neuron, one sub-sub-list per trial
                - trace_filenames: list of filenames of the read out .mat files
                - spiketimes_rec: list of lists of spiketimes, one sub-list per neuron, one sub-sub-list per trial
                - spiketimes_filenames: list of filenames of the read out .pkl files
    :rtype:     - traces_rec: list
                - trace_filenames: list
                - spiketimes_rec: list
                - spiketimes_filenames: list
    """

    # get a list of .mat filenames in traces directory
    if path_to_traces:
        path_to_traces = os.path.normpath(path_to_traces)
        try:
            trace_filenames = [f for f in os.listdir(path_to_traces)
                               if os.path.isfile(os.path.join(path_to_traces, f)) and f.endswith('.mat')]
            trace_filenames.sort()
        except FileNotFoundError:
            print('ERROR: ' + path_to_traces + ' was not found.')
            raise
        except:
            print('ERROR: ' + path_to_traces + ' was found but something went wrong trying to read files from it.')
            raise

    # get a list of .pkl filenames in spiketimes directory
    if path_to_spiketimes:
        path_to_spiketimes = os.path.normpath(path_to_spiketimes)
        try:
            spiketimes_filenames = [f for f in os.listdir(path_to_spiketimes)
                                    if os.path.isfile(os.path.join(path_to_spiketimes, f)) and f.endswith('.pkl')]
            spiketimes_filenames.sort()
        except FileNotFoundError:
            print('ERROR: ' + path_to_spiketimes + ' was not found.')
            raise
        except:
            print('ERROR: ' + path_to_spiketimes + ' was found but something went wrong trying to read files from it.')
            raise

    # get traces from files
    if path_to_traces:
        traces_rec = []
        for filename in trace_filenames:
            traces_rec.append(p_io.load_traces_from_mat(os.path.join(path_to_traces, filename)))
    else:
        traces_rec = None
        trace_filenames = None

    # get spiketimes from files
    if path_to_spiketimes:
        spiketimes_rec = []
        for filename in spiketimes_filenames:
            spiketimes_rec.append(p_io.load_spiketimes_from_pkl(os.path.join(path_to_spiketimes, filename)))
    else:
        spiketimes_rec = None
        spiketimes_filenames = None

    return traces_rec, trace_filenames, spiketimes_rec, spiketimes_filenames


def get_suppression_window(config_jam_only, spikes_jam_ft, config_jam_ft, info_jam_ft):
    # time window after playback onset in which a burst would be suppressed by jamming inhibition
    idx_nrn_oi_mod_abs_trc = p_util.get_abs_from_rel_nrn_idx(config_jam_ft['plot']['idx_nrn_oi_relative'],
                                                             config_jam_ft['plot']['idx_pop_oi'],
                                                             info_jam_ft[0]['population_sizes'])
    idx_t_start = 1
    idx_pop_fp = config_jam_only['plot']['idx_synpop_oi_for_fp']
    lat_playback_to_t_start = config_jam_only['input_current']['t_start'][idx_pop_fp][idx_t_start] \
                              - config_jam_only['misc']['playback_start']
    t_first_spike_no_jam = spikes_jam_ft[-1].t[np.where(spikes_jam_ft[-1].i == idx_nrn_oi_mod_abs_trc)[0][0]] / b2.ms
    t_unperturbed_bursts = []
    t_perturbed_bursts = []
    for mon in range(len(spikes_jam_ft)):
        t_start_jam = info_jam_ft[mon]['free_parameter_values'][idx_pop_fp][idx_t_start]
        playback_onset = t_start_jam - lat_playback_to_t_start
        idx_spikes = np.where(spikes_jam_ft[mon].i == idx_nrn_oi_mod_abs_trc)[0]
        t_unperturbed_bursts.append(t_first_spike_no_jam - playback_onset)  # relative to playback onset
        if np.any(spikes_jam_ft[mon].t[idx_spikes]):
            t_perturbed_bursts.append(spikes_jam_ft[mon].t[idx_spikes[0]] / b2.ms - playback_onset)
        else:
            t_perturbed_bursts.append(np.nan)
    stepsize_burst_time = config_jam_ft['free_parameter_stepsize']['t_start']
    i_suppressed_burst = np.where(np.isnan(t_perturbed_bursts))[0]
    original_burst_times = np.array(t_unperturbed_bursts)
    t_suppressed_bursts = original_burst_times[i_suppressed_burst]
    suppression_window_playback_ms_lr = [np.min(t_suppressed_bursts) - stepsize_burst_time / 2,
                                         np.max(t_suppressed_bursts) + stepsize_burst_time / 2]
    suppr_win_l = suppression_window_playback_ms_lr[0]
    suppr_win_r = suppression_window_playback_ms_lr[1]

    return [suppr_win_l, suppr_win_r]
