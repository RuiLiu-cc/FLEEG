from scipy.io import loadmat, savemat
import os
import resampy
import numpy as np
import scipy.signal as signal



##======ShanghaiU Dataset========
def shu_fetchfiles(dataPath, chans=None, downsampleFactor=None):
    '''
    Load one subject EEG data from the d1.mat~d5.mat files
    Param:
        dataPath: str, data path
        chans: list, channels to be loaded, default: None
        downsampleFactor: int, data down-sampling factor, default: None
    Returns:
        x: nparray, size:[trials x channels x times], data
        y: nparray, size: [trails], label
        s: int, size: 1, frequency
    '''

    # read the mat file:
    filename = dataPath + '/d1.mat'
    data = loadmat(filename)
    x = data['data']
    y = data['labels']
    s = 250
    del data

    for i in range(2,6):
        filename = dataPath + '/d{}.mat'.format(i)
        data = loadmat(filename)
        x = np.concatenate((x, data['data']), axis = 0)
        y = np.concatenate((y, data['labels']), axis = 1)
        del data
    # reshape the label into [#trials]
    y = np.squeeze(y)
    y = y-1

    # extract the requested channels:
    if chans is not None:
        x = x[:, np.array(chans), :]

    # down-sample if requested:
    if downsampleFactor is not None:
        xNew = np.zeros((x.shape[0], x.shape[1], int(x.shape[2]/downsampleFactor)), np.float32)
        for i in range(x.shape[1]): # resampy.resample cant handle the 3D data.
            xNew[:,i,:] = resampy.resample(x[:,i,:], s, s/downsampleFactor, axis=1)
        x = xNew
        s = s/downsampleFactor

    return x, y, s


## filter the signal with band-pass filter
def bandpassfilter_cheby2_sos(data, bandFiltCutF, fs, filtAllowance=[0.2,5], axis=2):
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



def preprocessing_shu_dataset(OrigDataPath, saveNpzPath, chans, downdampleFactor, bandFiltCutF, filterType, resample):

    # check if the filtered EEG .npz file exists or filter it.
    FilteredPath = os.path.join(saveNpzPath, '{}Hz_{}Hz_{}'.format(bandFiltCutF[0], bandFiltCutF[1], filterType))
    print('Processed data be saved in folder : ' + FilteredPath)

    if not os.path.exists(FilteredPath):
        os.makedirs(FilteredPath)

        # load the EEG data from the .mat file one subject by one subject
        subjects = list(range(1, 26))
        for iSub in subjects:
            print('Processing subject No.: {}'.format(iSub))
            x, y_data, fs = shu_fetchfiles(os.path.join(OrigDataPath, 'S{}'.format(iSub)),
                                 chans=chans, downsampleFactor=downdampleFactor)
            # filter the EEG signal
            if filterType == 'cheby2_sos':
                x_data = bandpassfilter_cheby2_sos(x, bandFiltCutF, fs)

            # resample the EEG signal
            if resample:
                x_data = signal.resample_poly(x_data, 5, 4, axis=2, padtype='constant')

            # save the preprocessed EEG signal
            filename = os.path.join(FilteredPath, 's{}.npz'.format(iSub))
            np.savez(filename, x_data=x_data, y_data=y_data)


if __name__ == '__main__':

    # filter the raw data and store the filtered EEG as array into a .np file for each subject
    OrigDataPath = r'/path/to/Original_data/ShanghaiU_MI'
    saveNpzPath = r'/path/to/filtered_data/ShanghaiU_MI'


    chans = None
    downdampleFactor = 1
    # chans = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
    # downsampleFactor = 4
    resample = False

    bandFiltCutF = [0.3, 40]
    filterType = 'cheby2_sos'
    preprocessing_shu_dataset(OrigDataPath, saveNpzPath, chans, downdampleFactor, bandFiltCutF, filterType, resample)