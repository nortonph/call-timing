"""Various data analysis functions, including spike detection and analysis of spike data. [call-time-model]
"""

import numpy as np
import brian2 as b2
from types import SimpleNamespace
from scipy.signal import savgol_filter

# import my functions
from fcn import p_io, p_util


def get_spiketimes(traces, sampling_frequency_khz=40, smooth_filter_win_len=11, threshold_factor=14,
                   max_ms_thresh_to_peak=1, min_samp_above_thresh=2):
    """Detect spiketimes in voltage traces by taking derivative (gradient) of the smoothed trace first.
    Smoothing uses Savitzky-Golay filter.

    :param traces: (list of) numpy ndarray(s) of voltage traces
    :type traces: np.ndarray or list
    :param sampling_frequency_khz: [default=40] sampling frequency of traces in kHz
    :type sampling_frequency_khz: int or float
    :param smooth_filter_win_len: [default=11] length of window for Savitzky-Golay filter
    :type smooth_filter_win_len: int
    :param threshold_factor: [default=14] gradient of smoothed trace must cross this factor times the estimated standard
        deviation of the noise in the absolute values of the gradient for a spike to be detected
        (see Quian Quiroga et al. (2004), doi:10.1162/089976604774201631)
    :type threshold_factor: int or float
    :param max_ms_thresh_to_peak: [default=1] spike peak has to be within this number of milliseconds after threshold
        crossing of the gradient. Otherwise it will be considered an artifact
    :type max_ms_thresh_to_peak: int or float
    :param min_samp_above_thresh: [default=2] gradient must be above threshold for this many consecutive samples to
        not be considered an artifact
    :type min_samp_above_thresh: int
    :return:
        - spiketimes_samp: list of lists of spike times (in samples). One sublist per trace.
        - b_spike_matrix: two-dimensional matrix of samples (columns) of traces (rows), where 0 == no spike & 1 == spike
        - thresholds: voltage threshold per trace for spike detection: trace crosses this from below prior to spike peak
    :rtype:
        - spiketimes_samp: [[int]]
        - b_spike_matrix: np.ndarray
        - thresholds: [float]
    """

    # if traces contains a single trace, convert it to a list
    if type(traces) is not list:
        traces = [traces]

    # check inputs
    assert all(isinstance(item, np.ndarray) for item in traces), \
        "traces must be a numpy ndarray or list of ndarrays"

    # initialize variables
    n_traces = len(traces)
    n_values = [traces[t].size for t in range(n_traces)]
    spiketimes_samp = []
    b_spike_matrix = np.zeros((n_traces, np.max(n_values)))
    thresholds = []

    # smooth traces using savitzky-golay filter
    traces_smooth_sg = []
    for trc in range(n_traces):
        traces_smooth_sg.append(savgol_filter(traces[trc], smooth_filter_win_len, 3))

    # calculate the derivative (gradient) of each smoothed trace and set negatives value to zero (rectify)
    gradients = []
    gradients_rect = []
    for trc in range(n_traces):
        gradients.append(np.gradient(traces_smooth_sg[trc]))
        gradients_rect.append(np.zeros((len(gradients[trc]), 1)).squeeze())
        gradients_rect[trc][gradients[trc] >= 0] = gradients[trc][gradients[trc] >= 0]

    # loop through traces and send them to detector
    for trc in range(n_traces):
        # set threshold
        noise_std_estimation = np.median(np.abs(gradients[trc]) / 0.6745)
        threshold_cur = threshold_factor * noise_std_estimation

        # get spiketimes
        trace_thresh_zero = gradients_rect[trc] - threshold_cur
        b_trace_above_thresh = (trace_thresh_zero > 0).astype(int)
        b_crosses_thresh = b_trace_above_thresh[1:] - b_trace_above_thresh[:-1]
        i_crossings_up = np.where(b_crosses_thresh > 0)[0]
        i_crossings_down = np.where(b_crosses_thresh < 0)[0]
        # continue analysis only if there is at least one upward and one downward crossing of the threshold
        if i_crossings_down.any() and i_crossings_up.any():
            # if first down crossing comes before first up (i.e. trace started
            # above threshold), delete that. And vice versa.
            if i_crossings_down[0] < i_crossings_up[0]:
                i_crossings_down = np.delete(i_crossings_down, 0, axis=0)
            if i_crossings_up[-1] > i_crossings_down[-1]:
                i_crossings_up = np.delete(i_crossings_up, -1, axis=0)
            # loop through threshold crossings and get index of maximum gradient
            i_max_g = []
            i_max_v = []
            for c in range(len(i_crossings_up)):
                # make sure gradient stays above threshold for at least 2
                # samples (to exclude crossings due to noise)
                if i_crossings_down[c] - i_crossings_up[c] >= min_samp_above_thresh:
                    i_max_above_thresh = np.argmax(gradients_rect[trc][i_crossings_up[c]:i_crossings_down[c]])
                    i_max_g.append(i_max_above_thresh + i_crossings_up[c])
                    # find first zero crossing after gradient maximum (i.e. timepoint of maximum smoothed voltage)
                    i_next_zero_crossings = np.squeeze(np.where(gradients[trc][i_max_g[-1]:] <= 0))
                    if i_next_zero_crossings.any():
                        if i_next_zero_crossings[0] <= max_ms_thresh_to_peak * sampling_frequency_khz:
                            # check if next zero crossing was already detected with previous threshold crossing and if
                            # so, ignore this one (otherwise there will be two spikes at detected at the same timepoint)
                            if i_max_v and not i_max_g[-1] + i_next_zero_crossings[0] - 1 == i_max_v[-1]:
                                i_max_v.append(i_max_g[-1] + i_next_zero_crossings[0] - 1)
                            elif not i_max_v:
                                i_max_v.append(i_max_g[-1] + i_next_zero_crossings[0] - 1)
                            else:
                                print(" double detection of one spike avoided at trace " + str(trc) + " spike " +
                                      str(len(i_max_v)) + "(?)")

                        else:
                            print(" found and ignored peak longer than max_ms_thresh_to_peak=" +
                                  str(max_ms_thresh_to_peak) + " after threshold crossing of gradient at " +
                                  str(i_max_g[-1] / sampling_frequency_khz) + "ms in trace " + str(trc) + ".")
            b_spike_matrix[trc, i_max_v] = 1
            spiketimes_samp.append(i_max_v)
        else:
            spiketimes_samp.append([])

        # save threshold
        thresholds.append(threshold_cur)

    return spiketimes_samp, b_spike_matrix, thresholds


def get_onset_of_spikes(traces, spiketimes_ms, sampling_frequency_khz=40, thresh_ratio_to_derivative_peak=0.066,
                        smooth_filter_win_len=11):
    """Given voltage trace(s) and corresponding spiketimes (i.e. voltage peaks), determine onsets of each spike (and
    spike thresholds) by detecting threshold crossing of the derivative of the smoothed membrane voltage,
    backward from the spike.

    :param traces: (list of) numpy ndarray(s) of voltage traces
    :type traces: np.ndarray or list
    :param spiketimes_ms: list of lists of spike times of the neuron (one sublist per trial, i.e. array in traces)
    :type spiketimes_ms: [[float]]
    :param sampling_frequency_khz: [default=40] sampling frequency of traces in kHz
    :type sampling_frequency_khz: int or float
    :param thresh_ratio_to_derivative_peak: threshold for the prespike membrane voltage derivative for spike onset /
        spike threshold detection. this is the ratio of the derivative peak at which the spike is considered to be
        triggered. See e.g. Azouz & Gray (2000), Trinh et al. (2019). See also Sekerli et al. (2004) for a comparison
        of detection methods and this and Fontaine et al. (2014) for an absolute threshold instead. Exchange
        thresh_derivative_mV_per_samp_relative with thresh..._absolute in code below for absolute threshold.
    :type thresh_ratio_to_derivative_peak: float
    :param smooth_filter_win_len: [default=11] length of window for Savitzky-Golay filter
    :type smooth_filter_win_len: int
    :return:    - spike_onset_times_ms: list of lists of spike onset times in milliseconds
                - spike_thresholds_mV: list of lists of spike thresholds (voltages at corresponding spike onsets)
    :rtype:     - spike_onset_times_ms: list
                - spike_thresholds_mV: list
    """

    # check inputs
    if type(traces) is not list:
        traces = [traces]
    assert len(traces) == len(spiketimes_ms), "arguments traces and spiketimes_ms must have the same number of " \
                                              "elements (sublists/arrays)"

    # initialize variables
    n_traces = len(traces)

    # smooth traces using savitzky-golay filter
    traces_smooth_sg = []
    for trc in range(n_traces):
        traces_smooth_sg.append(savgol_filter(traces[trc], smooth_filter_win_len, 3))

    # calculate the derivative (gradient) of each smoothed trace and set negatives value to zero (rectify)
    gradients = []
    for trc in range(n_traces):
        gradients.append(np.gradient(traces_smooth_sg[trc]))

    # loop through traces and spikes
    spike_onset_times_ms = []
    spike_thresholds_mV = []
    for trc in range(n_traces):
        spike_onset_times_ms.append([])
        spike_thresholds_mV.append([])
        for t_spike in spiketimes_ms[trc]:
            # find derivative maximum in the 1ms before spike
            samp_spike = int(round(t_spike * sampling_frequency_khz))
            samp_1ms_prespike = samp_spike - int(round(1 * sampling_frequency_khz))
            if samp_1ms_prespike < 0:
                print('! Onset calculation for spike within first ms skipped in trace ' + str(trc))
                spike_onset_times_ms[trc].append(None)
                spike_thresholds_mV[trc].append(None)
                break
            gradient_peak = np.max(gradients[trc][samp_spike:samp_1ms_prespike:-1])
            samp_gradient_peak_from_spike = np.argmax(gradients[trc][samp_spike:samp_1ms_prespike:-1])
            samp_gradient_peak = samp_spike - samp_gradient_peak_from_spike
            # find first threshold corssing of gradient backwards from peak
            thresh_derivative_mV_per_samp_absolute = 25 / sampling_frequency_khz
            # threshold relative to gradient peak
            thresh_derivative_mV_per_samp_relative = gradient_peak * thresh_ratio_to_derivative_peak
            samp_1ms_prepeak = samp_gradient_peak - int(round(1 * sampling_frequency_khz))
            thresh_crossings = np.nonzero(gradients[trc][samp_gradient_peak:samp_1ms_prepeak:-1]
                                          < thresh_derivative_mV_per_samp_relative)[0]
            if thresh_crossings.any():
                samp_thresh_crossing_from_peak = thresh_crossings[0]
                samp_spike_onset = samp_gradient_peak - samp_thresh_crossing_from_peak
                spike_onset_times_ms[trc].append(samp_spike_onset / sampling_frequency_khz)
                spike_thresholds_mV[trc].append(traces_smooth_sg[trc][samp_spike_onset])
            else:
                spike_onset_times_ms[trc].append(None)
                spike_thresholds_mV[trc].append(None)
                print("INFO: Spike onset not detectable")

    return spike_onset_times_ms, spike_thresholds_mV


def get_rise_time(trace, i_peak, i_onset, percent=100, sampling_frequency_khz=40):
    """Given the index of a peak (e.g. PSP or spike) in a trace and the index of the onset of the rise (i.e. spike
    or PSP onset), return the time in ms it takes to reach a certain amplitude percentage from baseline (i.e. value of
    trace at onset, e.g. in mV). Also return the index in the trace that corresponds to this timepoint.
    percent == 100: peak amplitude

    :param trace: numpy ndarray of e.g. voltage trace
    :type trace: np.ndarray
    :param i_peak: index of the peak of interest in trace
    :type i_peak: int
    :param i_onset: index of the onset of the event of interest in trace (e.g. PSP or spike onset)
    :type i_onset: int
    :param percent: [default=100] percentage of peak amplitude relative to baseline (amplitude at i_onset) to reach
    :type percent: int or float
    :param sampling_frequency_khz: [default=40] sampling frequency of traces in kHz
    :type sampling_frequency_khz: int or float
    :return:    - t_rise: time from onset to reach x percent of the peak amplitude in milliseconds
                - i_percent: index to trace of the timepoint when trace reaches x percent of the peak
    :rtype:     - t_rise: float
                - i_percent: int
    """

    # check inputs
    assert i_onset < i_peak < len(trace), "get_rise_time(): condition [i_onset < i_peak < len(trace)] not met"

    # if 100 percent rise time, just return the time difference between peak and onset
    if percent == 100:
        t_rise = (i_peak - i_onset) / sampling_frequency_khz
        i_percent = i_peak
        return t_rise, i_percent

    # get amplitude of interest (x percent of the difference between peak and onset)
    peak_amp = trace[i_peak]
    onset_amp = trace[i_onset]
    amp_at_x_perc = onset_amp + (peak_amp - onset_amp) * (percent / 100)

    # get timepoint where amplitude first exceeds the percentage after onset
    if peak_amp > onset_amp:
        i_percent_from_onset = np.nonzero(trace[i_onset:i_peak+1] >= amp_at_x_perc)[0][0]
    else:
        i_percent_from_onset = np.nonzero(trace[i_onset:i_peak+1] <= amp_at_x_perc)[0][0]

    # if the amplitude of interest is closer to the sample before the crossing, use that instead
    # (i.e. round rise time to nearest sample)
    if abs(amp_at_x_perc - trace[i_percent_from_onset]) > abs(amp_at_x_perc - trace[i_percent_from_onset-1]):
        i_percent_from_onset = i_percent_from_onset - 1

    # get time from onset to reach that point
    t_rise = i_percent_from_onset / sampling_frequency_khz
    i_percent = i_onset + i_percent_from_onset

    return t_rise, i_percent


def remove_spikes(filename_pkl, trace_nrs, spike_nrs):
    """Opens .pkl file that contains spiketimes of a recording, deletes the spiketimes given by the trace and spike
    indices, overwrites the .pkl file with the modified spiketimes. Call manually to remove false positives.

    :param filename_pkl: filename of .pkl file containing spiketimes
    :type filename_pkl: str
    :param trace_nrs: list of integer indices to traces in which spikes are to be deleted. 1-based indexing, i.e. [1]
        refers to the first trace and therefore to spiketimes[0]
    :type trace_nrs: [int]
    :param spike_nrs: list of list of indices to the chronologically ordered spikes. one sublist per trace in i_traces.
        1-based indexing. Example: remove_spikes('bla.pkl', [1, 3], [[1, 2], [1, 2, 3]]) removes the first two spikes in
        the first trace and the first three spikes in the third trace, i.e. spiketimes[0][0:2] and spiketimes[2][0:3]
    :type spike_nrs: [[int]]
    :return:
        - spiketimes_ms_modified: the list of spiketimes with spikes removed
        - spiketimes_ms_loaded: the original (unmodified) list of spiketimes (in milliseconds) from the .pkl file
    :rtype:
        - spiketimes_ms_modified: [[float]]
        - spiketimes_ms_loaded: [[float]]
    """

    # load spiketimes from pickle file
    spiketimes_ms_loaded = p_io.load_spiketimes_from_pkl(filename_pkl)

    # check inputs
    assert type(trace_nrs) is list and all(isinstance(item, int) and item > 0 for item in trace_nrs), \
        "trace_nrs must be a list of integer indices to traces in which spikes are to be deleted (1-based indexing)."
    assert type(spike_nrs) is list and all(isinstance(item, list) and item[0] > 0 for item in spike_nrs), \
        "spike_nrs must be a list of lists of integer indices to spikes, one sublist per trace (1-based indexing)."
    assert len(spike_nrs) is len(trace_nrs), "spike_nrs must have the same number of sublists as trace_nrs has elements"

    # remove spikes
    spiketimes_ms_modified = spiketimes_ms_loaded
    for count, trc in enumerate(trace_nrs):
        i_trc = trc - 1
        # check if indices are correct
        assert 0 < trc <= len(spiketimes_ms_loaded), "trace index " + str(trc) \
            + " is not within 1 and length of loaded spiketimes (" + str(len(spiketimes_ms_loaded)) + ")."
        assert all([i <= len(spiketimes_ms_loaded[i_trc]) for i in spike_nrs[count]]), \
            "spike index not found for trace " + str(trc) + ". length of spiketimes sublist for this trace = " \
            + str(len(spiketimes_ms_loaded[trc]))
        # convert indices from 1-based to 0-based
        i_spikes_to_remove = [i - 1 for i in spike_nrs[count]]
        # overwrite spiketimes_ms_modified, excluding those indexed to be removed
        spiketimes_ms_modified[i_trc] = \
            [spiketimes_ms_loaded[i_trc][s] for s in range(len(spiketimes_ms_loaded[i_trc]))
             if s not in i_spikes_to_remove]
        print("removed " + str(len(spike_nrs[count])) + " spikes from trace number " + str(trc) + " in " + filename_pkl)

    # overwrite pickle file with modified spiketimes
    p_io.save_spiketimes_to_pkl(spiketimes_ms_modified, filename_pkl)

    return spiketimes_ms_modified, spiketimes_ms_loaded


def get_psth(spiketimes_ms, duration_ms, t_start_ms=0, psth_dt_ms=5, b_smoothed_psth=True, smooth_filter_win_len=9):
    """Calculate peri-stimulus time histogram from spiketimes over several aligned trials or neurons. Returns bin edges
    in milliseconds and average local spike rate in Hertz.

    :param spiketimes_ms: list of lists of spike times of the recorded neurons (one sublist per neuron or trial)
    :type spiketimes_ms: [[float]]
    :param duration_ms: total duration of recording / simulation and therefore of returned psth
    :type duration_ms: int or float
    :param t_start_ms: (optional) start time of recording / simulation and therefore of returned psth. default: 0
    :type t_start_ms: int or float
    :param psth_dt_ms: [default=5] time bin in milliseconds for the peri-stimulus-time histogram (psth)
    :type psth_dt_ms: int or float
    :param b_smoothed_psth: [default=True] also return smoothed psth (using Savitzky-Golay filter)
    :type b_smoothed_psth: bool
    :param smooth_filter_win_len: [default=9] length of window for Savitzky-Golay filter
    :type smooth_filter_win_len: int
    :return:
        - t_values: list of time values of psth bins (left edges) in milliseconds
        - psth: list of estimated local spike rates in Hertz per time bin
        - psth_smooth: same as psth, but smoothed using Savitzky-Golay filter
    :rtype:
        - t_values: [float]
        - psth: [float]
        - psth_smooth: [float]
    """

    # calculate peri-stimulus-time histogram (psth)
    bin_edges_left = p_util.integer_linspace(t_start_ms, t_start_ms + duration_ms, psth_dt_ms, b_include_end=False)
    n_bins = len(bin_edges_left)
    psth = np.zeros(n_bins)
    i = 0  # index for psth throughout the loop
    t_values = []  # time values of bins (left value of each bin)
    spiketimes_ms_all = np.concatenate(spiketimes_ms)
    for t in bin_edges_left:
        n_spikes_in_bin = sum(np.logical_and(spiketimes_ms_all > t, spiketimes_ms_all < t + psth_dt_ms))
        psth[i] = n_spikes_in_bin / (psth_dt_ms * len(spiketimes_ms)) * 1000
        t_values.append(t + psth_dt_ms / 2)
        i += 1

    # get smoothed psth
    if b_smoothed_psth and any(spiketimes_ms_all):
        psth_smooth = savgol_filter(psth, smooth_filter_win_len, 3).tolist()
    else:
        if not b_smoothed_psth:
            psth_smooth = None
        else:
            psth_smooth = np.zeros(np.ceil(duration_ms / psth_dt_ms).astype(int)).tolist()
            print(' get_psth(): no spiketimes found -> psth_smooth set to all zeros')

    return t_values, psth.tolist(), psth_smooth


def sum_psths(t_values, psths, offset_ms=None, b_smoothed_psth=True, smooth_filter_win_len=9):
    """Sums up multiple psths and returns the sum, as well as (optionally) a smoothed version. All input psths must have
    the same bin size and all offsets (if any) must be an integer multiple of the bin size. If the psths have different
    alignments, supplying a list of offset values (offset_ms) will shift the psths by this amount prior to summation.

    :param t_values: list of lists of time values in milliseconds of the psth bins (left side).
    :type t_values: [[int]] or [[float]]
    :param psths: list of lists of non-smoothed psth values (average spike rate as returned by p_analysis.get_psth())
    :type psths: [[float]]
    :param offset_ms: [default=None] list of time values at which to align each psth (0 will be at this timepoint)
    :type offset_ms: [int] or [float]
    :param b_smoothed_psth: [default=True] also return smoothed psth (using Savitzky-Golay filter)
    :type b_smoothed_psth: bool
    :param smooth_filter_win_len: [default=9] length of window for Savitzky-Golay filter
    :type smooth_filter_win_len: int
    :return:
        - t_val_for_all: list of time values in milliseconds of the psth bins of the summed psth (left side).
        - psth_summed: summed psth
        - psth_summed_smoothed: summed and smoothed psth
        - t_values_offset: t_values aligned to offset_ms
    :rtype:
        - t_val_for_all: [int] or [float]
        - psth_summed: [float]
        - psth_summed_smoothed: [float]
        - t_values_offset: [[int]] or [[float]]
    """

    # check inputs
    if offset_ms is not None:
        assert len(t_values) is len(psths) is len(offset_ms), \
            "t_values, psths and offset_ms must all have the same number of elements"
    else:
        assert len(t_values) is len(psths), "t_values and psths must have the same number of elements"

    # align timepoints to offset
    t_values_offset = []
    if offset_ms is not None:
        for n in range(len(t_values)):
            t_values_offset.append([t_values[n][i] - offset_ms[n] for i in range(len(t_values[n]))])

    # if a single psth was given, just return these values, after offsetting the t_values
    if len(psths) == 1:
        psth_smoothed = savgol_filter(psths[0], smooth_filter_win_len, 3)
        return t_values_offset[0], psths[0], psth_smoothed.tolist(), t_values_offset

    # get intersection of bin edge time values of all psths
    t_val_for_all = np.intersect1d(t_values_offset[0], t_values_offset[1])
    for i in p_util.integer_linspace(2, len(t_values_offset)-1, 1, b_include_end=True):
        t_val_for_all = np.intersect1d(t_val_for_all, t_values_offset[i])

    # sum psth values at those time points
    psth_summed = np.zeros(len(t_val_for_all))
    for t in range(len(t_val_for_all)):
        for p in range(len(t_values_offset)):
            idx_t_cur = t_values_offset[p].index(t_val_for_all[t])
            psth_summed[t] = psth_summed[t] + psths[p][idx_t_cur]

    # get smoothed psth
    if b_smoothed_psth:
        psth_summed_smoothed = savgol_filter(psth_summed, smooth_filter_win_len, 3)
    else:
        psth_summed_smoothed = None

    return t_val_for_all.tolist(), psth_summed.tolist(), psth_summed_smoothed.tolist(), t_values_offset


def subthreshold_wiggliness(statemon, spikemon, info, idx_nrn_abs, t_start, t_end):
    """Returns the standard deviation of the membrane potential derivative of a neuron over a given timespan.

    :param statemon: brian2 StateMonitor - like(!) SimpleNamespace from file or dict from b2...get_states()
    :type statemon: SimpleNamespace or dict
    :param spikemon: brian2 SpikeMonitor - like(!) SimpleNamespace from file or dict from b2...get_states()
    :type spikemon: SimpleNamespace or dict
    :param info: dictionary containing additional information about simulation
    :type info: dict
    :param idx_nrn_abs: index of neuron to be analyzed within statemon
    :type idx_nrn_abs: int
    :param t_start: beginning of timespan to be analyzed in milliseconds
    :type t_start: int or float
    :param t_end: end of timespan to be analyzed in milliseconds
    :type t_end: int or float
    :return: std_diff_v: standard deviation of membrane potential derivative
    :rtype: std_diff_v: float
    """

    # check inputs
    assert isinstance(statemon, b2.monitors.statemonitor.StateMonitor) or isinstance(statemon, SimpleNamespace), \
        "statemon must be a b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace object"
    assert isinstance(spikemon, b2.monitors.statemonitor.StateMonitor) or isinstance(spikemon, SimpleNamespace), \
        "spikemon must be a b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace object"
    assert isinstance(t_start, int) or isinstance(t_start, float) and \
        isinstance(t_end, int) or isinstance(t_end, float) and t_start >= 0 and t_end >= 0 and t_end > t_start, \
        "t_start and t_end must be positive integers or floats with t_end > t_start"

    # if there is no t_end, return nan
    if np.isnan(t_end):
        return np.nan

    # if neuron_index is a negative number (i.e. counting from end), translate that to a positive index
    if idx_nrn_abs < 0:
        idx_nrn_abs = sum(info['population_sizes']) + idx_nrn_abs

    # get indices to t_start and t_end
    idx_start = np.where(statemon.t / b2.ms >= t_start)[0][0]
    idx_end = np.where(statemon.t / b2.ms >= t_end)[0][0]

    # get membrane potential derivative and number of spikes between t_start and t_end
    diff_v = np.diff(statemon.v[idx_nrn_abs, idx_start:idx_end] / b2.mV)
    i_spiketimes_nrn = np.where(spikemon.i == idx_nrn_abs)[0]
    if any(i_spiketimes_nrn):
        n_spikes = np.sum(np.logical_and(t_start < spikemon.t[i_spiketimes_nrn] / b2.ms,
                                         spikemon.t[i_spiketimes_nrn] / b2.ms < t_end))
    else:
        n_spikes = 0

    # sort derivative values and remove smallest n (largest negative slope), where n is number of spikes
    # (removing large slopes due to LIF reset)
    diff_v_cut = np.sort(diff_v)[range(n_spikes, len(diff_v))]

    # get standard deviation of membrane potential derivative (after spikes removed)
    std_diff_v = np.std(diff_v_cut)

    return std_diff_v


def get_average_potential(statemon, info, idx_nrn_abs, t_start, t_end):
    """Returns the average membrane potential of a neuron over a given timespan.

    :param statemon: brian2 StateMonitor or -like(!) SimpleNamespace from file
    :type statemon: SimpleNamespace or b2.monitors.statemonitor.StateMonitor
    :param info: dictionary containing additional information about simulation
    :type info: dict
    :param idx_nrn_abs: index of neuron to be analyzed within statemon
    :type idx_nrn_abs: int
    :param t_start: beginning of timespan to be analyzed in milliseconds
    :type t_start: int or float
    :param t_end: end of timespan to be analyzed in milliseconds
    :type t_end: int or float
    :return: avg_v: average of membrane potential
    :rtype: avg_v: float
    """

    # check inputs
    assert isinstance(statemon, b2.monitors.statemonitor.StateMonitor) or isinstance(statemon, SimpleNamespace), \
        "statemon must be a b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace object"
    assert isinstance(t_start, int) or isinstance(t_start, float) and \
        isinstance(t_end, int) or isinstance(t_end, float) and t_start >= 0 and t_end >= 0 and t_end > t_start, \
        "t_start and t_end must be positive integers or floats with t_end > t_start"

    # if there is no t_end, return nan
    if np.isnan(t_end):
        return np.nan

    # if neuron_index is a negative number (i.e. counting from end), translate that to a positive index
    if idx_nrn_abs < 0:
        idx_nrn_abs = sum(info['population_sizes']) + idx_nrn_abs

    # get indices to t_start and t_end
    idx_start = np.where(statemon.t / b2.ms >= t_start)[0][0]
    idx_end = np.where(statemon.t / b2.ms >= t_end)[0][0]

    # get average membrane potential between t_start and t_end
    avg_v = np.mean(statemon.v[idx_nrn_abs, idx_start:idx_end] / b2.mV)

    return avg_v


def get_stdev_potential(statemon, info, idx_nrn_abs, t_start, t_end):
    """Returns the standard deviation of the membrane potential of a neuron over a given timespan.

    :param statemon: brian2 StateMonitor or -like(!) SimpleNamespace from file
    :type statemon: SimpleNamespace or b2.monitors.statemonitor.StateMonitor
    :param info: dictionary containing additional information about simulation
    :type info: dict
    :param idx_nrn_abs: index of neuron to be analyzed within statemon
    :type idx_nrn_abs: int
    :param t_start: beginning of timespan to be analyzed in milliseconds
    :type t_start: int or float
    :param t_end: end of timespan to be analyzed in milliseconds
    :type t_end: int or float
    :return: stdev_v: standard deviation of membrane potential
    :rtype: stdev_v: float
    """

    # check inputs
    assert isinstance(statemon, b2.monitors.statemonitor.StateMonitor) or isinstance(statemon, SimpleNamespace), \
        "statemon must be a b2.StateMonitor or (if loaded through p_io.load_monitors) SimpleNamespace object"
    assert isinstance(t_start, int) or isinstance(t_start, float) and \
        isinstance(t_end, int) or isinstance(t_end, float) and t_start >= 0 and t_end >= 0 and t_end > t_start, \
        "t_start and t_end must be positive integers or floats with t_end > t_start"

    # if there is no t_end, return nan
    if np.isnan(t_end):
        return np.nan

    # if neuron_index is a negative number (i.e. counting from end), translate that to a positive index
    if idx_nrn_abs < 0:
        idx_nrn_abs = sum(info['population_sizes']) + idx_nrn_abs

    # get indices to t_start and t_end
    idx_start = np.where(statemon.t / b2.ms >= t_start)[0][0]
    idx_end = np.where(statemon.t / b2.ms >= t_end)[0][0]

    # get average membrane potential between t_start and t_end
    stdev_v = np.std(statemon.v[idx_nrn_abs, idx_start:idx_end] / b2.mV)

    return stdev_v


def get_t_first_spike_mod(spikemons, info, idx_nrn_abs, search_from_ms=0):
    """Get the timepoint in the simulation when neuron idx_nrn_abs fired its first spike.

    :param spikemons: (list of) brian2 SpikeMonitor - like(!) SimpleNamespaces from file or dicts from b2...get_states()
    :type spikemons: SimpleNamespace or dict or list
    :param info: dictionary or list of dicts containing additional information about simulation (one dict per run)
    :type info: dict or list
    :param idx_nrn_abs: absolute index to the neuron
    :type idx_nrn_abs: int
    :param search_from_ms: if greater than 0, look for the first spike after this time in milliseconds
    :type search_from_ms: int or float
    :return: spike_time: list of times of first spike of neuron given by idx_nrn_abs. One entry per monitor (run)
    :rtype: spike_time: list
    """

    # if spikemons and info contain a single entry, convert them to a list
    if type(spikemons) is not list:
        spikemons = [spikemons]
    if type(info) is not list:
        info = [info]

    # if idx_nrn_abs is a negative number, translate that to a positive index
    if idx_nrn_abs < 0:
        idx_nrn_abs = sum(info[0]['population_sizes']) + idx_nrn_abs

    spike_time = np.zeros(len(spikemons))
    for mon in range(len(spikemons)):
        # get time of first spike
        if search_from_ms > 0:
            idx_all_spikes_nrn_oi = np.where(spikemons[mon].i == idx_nrn_abs)[0]
            idx_spikes_after = np.where(spikemons[mon].t[idx_all_spikes_nrn_oi] / b2.ms > search_from_ms)[0]
            idx_spikes = idx_all_spikes_nrn_oi[idx_spikes_after]
        else:
            idx_spikes = np.where(spikemons[mon].i == idx_nrn_abs)[0]
        # convert values back to milliseconds
        if np.any(spikemons[mon].t[idx_spikes]):
            spike_time[mon] = spikemons[mon].t[idx_spikes[0]] / b2.ms
    # set values NaN where no spike occured
    spike_time[spike_time == 0] = np.nan

    return spike_time


def get_psp_count(spikemon, connectivity, info, idx_nrn_abs):
    """Get the number of excitatory and inhibitory post-synaptic potentials (i.e. presynaptic spikes separated by
    positive/negative synaptic weights) of a neuron (idx_nrn_abs).

    :param spikemon: brian2 SpikeMonitor - like(!) SimpleNamespace from file
    :type spikemon: SimpleNamespace
    :param connectivity: dictionary containing synaptic connectivity information
    :type connectivity: dict
    :param info: dictionary containing additional information about simulation
    :type info: dict
    :param idx_nrn_abs: index of neuron to be analyzed within statemon
    :type idx_nrn_abs: int
    :return:    - n_epsp: number of excitatory post synaptic potentials, i.e. presynaptic spikes in synapse groups that
                    have positive weights.
                - n_ipsp: number of inhibitory post synaptic potentials, i.e. presynaptic spikes in synapse groups that
                    have negative weights.
    :rtype:     - n_epsp: int
                - n_ipsp: int
    """

    # get relative idx and population idx of neuron of interest
    idx_nrn_rel, idx_pop_oi = p_util.get_rel_from_abs_nrn_idx(idx_nrn_abs, info['population_sizes'])

    # loop through synapse groups and get the number of spikes of all presynaptic neurons if pop_oi is post
    n_ipsp = n_epsp = 0
    for syn in range(len(info['syn_post_idx'])):
        if info['syn_post_idx'][syn] == idx_pop_oi:
            idx_nrn_pre_rel = connectivity[syn]['i'][connectivity[syn]['j'] == idx_nrn_rel]
            if info['syn_weight'][syn] > 0:
                n_spikes_per_idx_all = np.bincount(spikemon.i)
                n_epsp += np.sum(n_spikes_per_idx_all[idx_nrn_pre_rel])
            elif info['syn_weight'][syn] < 0:
                n_spikes_per_idx_all = np.bincount(spikemon.i)
                n_ipsp += np.sum(n_spikes_per_idx_all[idx_nrn_pre_rel])

    return int(n_epsp), int(n_ipsp)


def prespike_trace_average(traces, spiketimes_ms, sampling_frequency_khz=40, duration=40, smooth_filter_win_len=11,
                           b_normalize=True, b_spiketime_is_onset=False, b_smooth=True):
    """Calculate the average membrane potential of all smoothed traces of a neuron over the timecourse from duration
    before and up to the first spike onset in the the trace. Returns all smoothed time series (len=duration) and average

    :param traces: (list of) numpy ndarray(s) of voltage traces
    :type traces: np.ndarray or list
    :param spiketimes_ms: list of lists of spike times of the neuron (one sublist per trial, i.e. array in traces)
    :type spiketimes_ms: [[float]]
    :param sampling_frequency_khz: [default=40] sampling frequency of traces in kHz
    :type sampling_frequency_khz: int or float
    :param duration: [default=100] duration of the returned time series relative to the first spike in each trace,
        in ms. negative value: timespan after spike.
    :type duration: int or float
    :param smooth_filter_win_len: [default=11] length of window for Savitzky-Golay filter
    :type smooth_filter_win_len: int
    :param b_normalize: if True,, normalize returned timesieries so that final value (spike onset) is 0 mV
    :type b_normalize: bool or int
    :param b_spiketime_is_onset: if True, spiketimes_ms is considered onset of spike (e.g. LIF model neurons)
    :type b_spiketime_is_onset: bool or int
    :param b_smooth: if True, smooth traces first
    :type b_smooth: bool or int
    :return: - timeseries_average: average time series of all traces that contain a spike and in which the time series
                before/after the first spike fully lies within the duration of the trace
             - all_timeseries: list containing all timeseries that were included in the average (i.e. section of trace
                of length duration before/after the first spike in each trace)
             - n_traces_included: number of traces that were included in the average (i.e. that fit the above criteria)
    :rtype:  - timeseries_average: numpy.ndarray
             - all_timeseries: list
             - n_traces_included: int
    """

    # check inputs
    if type(traces) is not list:
        traces = [traces]
    assert len(traces) == len(spiketimes_ms), "arguments traces and spiketimes_ms must have the same number of " \
                                              "elements (sublists/arrays)"

    # smooth traces using savitzky-golay filter (from scipy.signal)
    traces_smooth_sg = []
    if b_smooth:
        for trc in range(len(traces)):
            traces_smooth_sg.append(savgol_filter(traces[trc], smooth_filter_win_len, 3))
    else:
        traces_smooth_sg = traces

    # loop through sublists of spiketimes_ms and collect time series of length duration before/after first spike from
    # each trace, if the trace contains a spike and the time series fully lies within the trace
    all_timeseries = []
    n_traces_included = 0
    for i_cur_trace, spiketimes_cur_trace in enumerate(spiketimes_ms):
        if spiketimes_cur_trace:
            t_first_spike = spiketimes_cur_trace[0]
            if b_spiketime_is_onset:
                t_spike_onset = t_first_spike
            else:
                t_spike_onset, spike_threshold = get_onset_of_spikes(traces[i_cur_trace], [[t_first_spike]],
                                                                     sampling_frequency_khz=sampling_frequency_khz)
                t_spike_onset = t_spike_onset[0][0]
                spike_threshold = spike_threshold[0][0]
            samp_spike_onset = int(np.ceil(t_spike_onset * sampling_frequency_khz))
            t_start_or_end = t_spike_onset - duration
            samp_start_or_end = samp_spike_onset - int(round(duration * sampling_frequency_khz))
            if t_start_or_end < t_spike_onset:
                if samp_start_or_end >= 0:
                    if b_normalize:
                        all_timeseries.append(traces_smooth_sg[i_cur_trace][samp_start_or_end:samp_spike_onset] -
                                              traces_smooth_sg[i_cur_trace][samp_spike_onset - 1])
                    else:
                        all_timeseries.append(traces_smooth_sg[i_cur_trace][samp_start_or_end:samp_spike_onset])
                    n_traces_included += 1
                else:
                    print("! Trace " + str(i_cur_trace) + " skipped, because first spike is less than time series " +
                          "duration (" + str(duration) + "ms) into the recording...")
            elif t_start_or_end > t_spike_onset:
                if samp_start_or_end < len(traces_smooth_sg):
                    if b_normalize:
                        all_timeseries.append(traces_smooth_sg[i_cur_trace][samp_spike_onset:samp_start_or_end] -
                                              traces_smooth_sg[i_cur_trace][samp_spike_onset - 1])
                    else:
                        all_timeseries.append(traces_smooth_sg[i_cur_trace][samp_spike_onset:samp_start_or_end])
                    n_traces_included += 1
                else:
                    print("! Trace " + str(i_cur_trace) + " skipped, because first spike is less than time series " +
                          "duration (" + str(duration) + "ms) from the end of the recording...")

    # calculate the average of all time series
    timeseries_average = np.mean(all_timeseries, axis=0)

    return timeseries_average, all_timeseries, n_traces_included


def check_conditions(spikemons, statemons, infos, config_lo, n_param_vals_x, n_param_vals_y,
                     baseline_cond, n_spikes_cond, t_baseline, t_spikes):
    """Check if the baseline membrane potential of the neuron of interest (defined in config_lo['plot']
    ['idx_nrn_oi_relative']) from a parameter exploration with two free parameters (or one parameter for two synapses/
    populations) is within a certain range, and if it fired a certain number of spikes within a certain timespan.

    :param spikemons: list of dict(s) containing SpikeMonitor data from b2...get_states()
    :type spikemons: list
    :param statemons: list of dict(s) containing StateMonitor data from b2...get_states()
    :type statemons: list
    :param infos: list of dict(s) containing additional information about simulation (as created by run_simulation())
    :type infos: list
    :param config_lo: config dict as loaded from .json file (output directory), contains lower limits of all params
    :type config_lo: dict
    :param n_param_vals_x: number of values for first free parameter
    :type n_param_vals_x: int
    :param n_param_vals_y: number of values for first free parameter
    :type n_param_vals_y: int
    :param baseline_cond: condition for baseline membrane potential. tuple of lower and upper limit for condition to
        be considered met.
    :type baseline_cond: tuple or list
    :param n_spikes_cond: condition for number of spikes. tuple of lower and upper limit for condition to
        be considered met.
    :type n_spikes_cond: tuple or list
    :param t_baseline: time window in which the baseline membrane potential condition was checked (start, end)
    :type t_baseline: tuple or list
    :param t_spikes: time window in which the number of spikes condition was checked (start, end)
    :type t_spikes: tuple or list
    :return:  conditions_met: numpy array of dimension c*x*y, where c is the number of conditions checked and x/y is a
        2d matrix of the space of the two varied parameters, in which 1=condition met, 0=not met.
    :rtype:   conditions_met: numpy.ndarray
    """

    # constants
    n_conditions = 2

    # set conditions
    conditions_met_linear = np.zeros((n_conditions, n_param_vals_x * n_param_vals_y))
    t1_baseline = t_baseline[0] * b2.ms
    t2_baseline = t_baseline[1] * b2.ms
    t1_spikes = t_spikes[0] * b2.ms
    t2_spikes = t_spikes[1] * b2.ms
    min_v_baseline = baseline_cond[0] * b2.mV
    max_v_baseline = baseline_cond[1] * b2.mV
    min_n_spikes = n_spikes_cond[0]
    max_n_spikes = n_spikes_cond[1]

    # check for conditions met and add a 1 to the output arrays
    for m in range(len(spikemons)):
        # get neuron of interest from config
        nrn_oi_abs = p_util.get_abs_from_rel_nrn_idx(config_lo['plot']['idx_nrn_oi_relative'],
                                                     config_lo['plot']['idx_pop_oi'],
                                                     infos[m]['population_sizes'])
        b_spike_from_nrn_oi = spikemons[m].i == nrn_oi_abs
        t_spike_from_nrn_oi = spikemons[m].t[b_spike_from_nrn_oi]
        n_spikes_in_win = sum(np.logical_and(t1_spikes <= t_spike_from_nrn_oi, t_spike_from_nrn_oi <= t2_spikes))
        i1 = np.where(statemons[m].t >= t1_baseline)[0][0]
        i2 = np.where(statemons[m].t >= t2_baseline)[0][0]
        baseline_pm = np.mean(statemons[m].v[nrn_oi_abs, i1:i2])

        if min_n_spikes <= n_spikes_in_win <= max_n_spikes:
            conditions_met_linear[0, m] = 1
        if min_v_baseline <= baseline_pm <= max_v_baseline:
            conditions_met_linear[1, m] = 1

    conditions_met = conditions_met_linear.reshape((n_conditions, n_param_vals_y, n_param_vals_x))

    return conditions_met
