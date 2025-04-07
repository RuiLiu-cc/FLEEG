import mne
from moabb.datasets import MunichMI
import resampy
import numpy as np
import os
import scipy.signal as signal



##======MunichMI Dataset========
def process_raw(raw, dataset, channels=None, baseline=None):
    # get the trails from the raw data
    # revised from the Moabb.BaseParadigm.process_raw function, refer to:
    # https://github.com/NeuroTechX/moabb/blob/214ae4916afb2ca0d8f79590c2c611cd210e4ade/moabb/paradigms/base.py#L67
    '''
    Process raw data file for one subject of one session: partition into trials.
    Params:
        raw: mne.RawArray, raw EEG data.
        dataset: dataset instance in moabb, contains specific information
        channels: list of str, default: None, list of channel to select. If None, use all EEG channels available in
        the dataset.
        baseline: tuple of length 2, default: None, The time interval to consider as “baseline” when applying baseline
            correction. If a tuple (a, b), the interval is between a and b (in seconds),including the endpoints.
            Correction is applied by computing the mean of the baseline period and subtracting it from the data (see mne.Epochs)
            If None, do not apply baseline correction.
    Return:
        x: np.ndarray, size: [#trials, #timesteps], partitioned EEG data.
        labels: np.ndarray, size: [#trials]
    '''

    tmin = 0
    tmax = None

    # get events id
    event_id = dataset.event_id

    # find the events, first check stim_channels then annotations
    stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
    if len(stim_channels) > 0:
        events = mne.find_events(raw, shortest_event=0, verbose=False)
    else:
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)

    # picks channels
    if channels is None:
        picks = mne.pick_types(raw.info, eeg=True, stim=False)
    else:
        picks = mne.pick_channels(raw.info["ch_names"], include=channels, ordered=True)

    # pick events, based on event_id
    events = mne.pick_events(events, include=list(event_id.values()))

    # get interval
    tmin = tmin + dataset.interval[0]
    if tmax is None:
        tmax = dataset.interval[1]
    else:
        tmax = tmax + dataset.interval[0]

    if baseline is not None:
        baseline = (
            baseline[0] + dataset.interval[0],
            baseline[1] + dataset.interval[0],
        )
        bmin = baseline[0] if baseline[0] < tmin else tmin
        bmax = baseline[1] if baseline[1] > tmax else tmax
    else:
        bmin = tmin
        bmax = tmax

    # epoch data
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=bmin,
        tmax=bmax,
        proj=False,
        baseline=baseline,
        preload=True,
        verbose=False,
        picks=picks,
        event_repeated="drop",
        on_missing="ignore",
    )

    if bmin < tmin or bmax > tmax:
        epochs.crop(tmin=tmin, tmax=tmax)
    # if resample is not None:
    #     epochs = epochs.resample(resample)

    # rescale to work with uV
    x = dataset.unit_factor * epochs.get_data()
    x = x[:,:,0:3500]
    # x = x[:, :, 0:1750]

    labels = np.array([e for e in epochs.events[:, -1]])

    return x, labels


def MunichMI_fetchfiles(saveNpzPath, chans=None):

    # download the data
    data_orig = MunichMI()
    for sub_idx in range(1,11):
        _ = data_orig.data_path(sub_idx)


    # get the EEG one by one
    fs = 500
    for sub_idx in range(1,11):
        sessions = data_orig._get_single_subject_data(sub_idx)
        metadata = []
        for session, runs in sessions.items():
            for run, raw in runs.items():
                x, y = process_raw(raw, data_orig, channels=chans)
                met = {}
                met["subject"] = sub_idx
                met["session"] = session
                met["run"] = run
                metadata.append(met)

                # concatenate all sessions
                if session == 'session_0':
                    x_data = x
                    y_data = y
                else:
                    x_data = np.append(x_data, x, axis=0)
                    y_data = np.append(y_data, y, axis=0)

        # save the raw data into .npz
        filename = os.path.join(saveNpzPath, 's{}.npz'.format(sub_idx))
        np.savez(filename, x_data=x_data, y_data=y_data, metadata=metadata, fs=fs)


## filter the signal with band-pass filter
def bandpassfilter_cheby2_sos(data, bandFiltCutF, fs, filtAllowance=[0.2, 5], axis=2):
    '''
    Band-pass filter the EEG signal of one subject using cheby2 IIR filtering
    and implemented as a series of second-order filters with direct-form II transposed structure.

    Param:
        data: nparray, size [trials x channels x times], original EEG signal
        bandFiltCutF: list, len: 2, low and high cut off frequency (Hz).
                If any value is None then only one-side filtering is performed.
        fs: sampling frequency (Hz)
        filtAllowance: list, len: 2, transition bandwidth (Hz) of low-pass and high-pass f
        axis: the axis along which apply the filter.
    Returns:
        data_out: nparray, size [trials x channels x times], filtered EEG signal
    '''

    aStop = 40  # stopband attenuation
    aPass = 1  # passband attenuation
    nFreq = fs / 2  # Nyquist frequency

    if (bandFiltCutF[0] == 0 or bandFiltCutF[0] is None) and (bandFiltCutF[1] == None or bandFiltCutF[1] >= fs / 2.0):
        # no filter
        print("Not doing any filtering. Invalid cut-off specifications")
        return data

    elif bandFiltCutF[0] == 0 or bandFiltCutF[0] is None:
        # low-pass filter
        print("Using lowpass filter since low cut hz is 0 or None")
        fPass = bandFiltCutF[1] / nFreq
        fStop = (bandFiltCutF[1] + filtAllowance[1]) / nFreq
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'lowpass', output='sos')

    elif (bandFiltCutF[1] is None) or (bandFiltCutF[1] == fs / 2.0):
        # high-pass filter
        print("Using highpass filter since high cut hz is None or nyquist freq")
        fPass = bandFiltCutF[0] / nFreq
        fStop = (bandFiltCutF[0] - filtAllowance[0]) / nFreq
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'highpass', output='sos')

    else:
        # band-pass filter
        # print("Using bandpass filter")
        fPass = (np.array(bandFiltCutF) / nFreq).tolist()
        fStop = [(bandFiltCutF[0] - filtAllowance[0]) / nFreq, (bandFiltCutF[1] + filtAllowance[1]) / nFreq]
        # find the order
        [N, wn] = signal.cheb2ord(fPass, fStop, aPass, aStop)
        sos = signal.cheby2(N, aStop, wn, 'bandpass', output='sos')

    # dataOut = signal.sosfilt(sos, data, axis=axis)
    dataOut = signal.sosfiltfilt(sos, data, axis=axis)

    return dataOut



def preprocessing_MunichMI_dataset(saveNpzPath, saveFilteredPath, chans, downsampleFactor, bandFiltCutF, filterType):

    # check if the original file exists or fetch it.
    if not os.path.exists(saveNpzPath):
        os.makedirs(saveNpzPath)
        MunichMI_fetchfiles(saveNpzPath, chans=chans)

    # check if the filtered EEG .npz file exists or filter it.
    FilteredPath = os.path.join(saveFilteredPath, '{}Hz_{}Hz_{}'.format(bandFiltCutF[0], bandFiltCutF[1], filterType))
    if not os.path.exists(FilteredPath):
        os.makedirs(FilteredPath)

        # load the EEG data from the .mat file one subject by one subject
        files = os.listdir(saveNpzPath)
        files = sorted(files)
        for filename in files:
            data_orig = np.load(os.path.join(saveNpzPath, filename))
            x = data_orig['x_data']
            y_data = data_orig['y_data']
            y_data = y_data-1
            fs = data_orig['fs']


            # filter the EEG signal
            if filterType == 'cheby2_sos':
                x = bandpassfilter_cheby2_sos(x, bandFiltCutF, fs)

            # resample the signal
            if downsampleFactor is not None:
                xNew = np.zeros((x.shape[0], x.shape[1], int(x.shape[2] / downsampleFactor)), np.float64)
                for i in range(x.shape[1]):  # resampy.resample cant handle the 3D data.
                    xNew[:, i, :] = resampy.resample(x[:, i, :], fs, fs / downsampleFactor, axis=1)
                x_data = xNew

            # save the preprocessed EEG signal
            filename_out = os.path.join(FilteredPath, filename)
            np.savez(filename_out, x_data=x_data, y_data=y_data)


if __name__ == '__main__':

    # filter the raw data and store the filtered EEG as array into a .np file for each subject
    saveNpzPath = r'/path/to/Original_data/MunichMI'
    saveFilteredPath = r'/path/to/filtered_data/MunichMI'

    chans = None
    downdampleFactor = 2

    bandFiltCutF = [0.3, 40]
    filterType = 'cheby2_sos'
    preprocessing_MunichMI_dataset(saveNpzPath, saveFilteredPath, chans, downdampleFactor, bandFiltCutF, filterType)