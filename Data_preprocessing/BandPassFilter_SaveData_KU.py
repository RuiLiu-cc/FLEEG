from scipy.io import loadmat, savemat
import os
import resampy
import numpy as np
import scipy.signal as signal


##======KoreaU Dataset========
def ku_fetchfiles(dataPath, chans=None):
    '''
    Load one subject EEG data from the EEG_MI.mat file
    Param:
        dataPath: str, data path
        chans: list, channels to be loaded, default: None
    Returns:
        a dictionary, contains:
            x: nparray, size:[trials x channels x times], data
            y: nparray, size: [trails], label
            c: nparray, size: [channels],channels
            s: int, size: 1, frequency
    '''

    # read the mat file:
    data = loadmat(dataPath)
    x = np.concatenate((data['EEG_MI_train'][0,0]['smt'], data['EEG_MI_test'][0,0]['smt']), axis = 1).astype(np.float32)
    y = np.concatenate((data['EEG_MI_train'][0,0]['y_dec'].squeeze(), data['EEG_MI_test'][0,0]['y_dec'].squeeze()), axis = 0).astype(int)-1
    c = np.array([m.item() for m in data['EEG_MI_train'][0,0]['chan'].squeeze().tolist()])
    s = data['EEG_MI_train'][0,0]['fs'].squeeze().item()
    del data

    # extract the requested channels:
    if chans is not None:
        x = x[:,:, np.array(chans)]
        c = c[np.array(chans)]

    # # down-sample if requested:
    # if downsampleFactor is not None:
    #     xNew = np.zeros((int(x.shape[0]/downsampleFactor), x.shape[1], x.shape[2]), np.float32)
    #     for i in range(x.shape[2]): # resampy.resample cant handle the 3D data.
    #         xNew[:,:,i] = resampy.resample(x[:,:,i], s, s/downsampleFactor, axis = 0)
    #     x = xNew
    #     s = s/downsampleFactor

    # change the data dimensions to be in a format: trials x channels x times
    x = np.transpose(x, axes = (1,2,0))

    return {'x': x, 'y': y, 'c': c, 's':s}

def ku_save_mats(dataPath, savePath, chans=None):
    '''
    Save all sessions of one subject in one .mat file.
    Param:
        dataPath: str, original data folder path
        savePath: str, save folder path
        chans: list, channels to be loaded, default: None
    Return:
        None
    '''
    subjects = list(range(1,55))
    subL = ['session1', 'session2']

    print('Processed data be saved in folder : ' + savePath)
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    for iSub in subjects:
        data_all = {}
        for iSession in subL:
            print('Processing subject No.: {}'.format(iSub))
            data = ku_fetchfiles(os.path.join(dataPath, iSession, 's{}'.format(iSub), 'EEG_MI.mat'),
                                chans=chans)
            data_all[iSession] = data
            data_all['SubjectNo'] = iSub
        savemat(os.path.join(savePath, 's{}.mat'.format(iSub)), data_all)



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



def preprocessing_ku_dataset(OrigDataPath, saveMatPath, saveNpzPath, chans, downsampleFactor, bandFiltCutF, filterType):

    # check if the allSession .mat file exists or fetch it.
    if not os.path.exists(saveMatPath):
        ku_save_mats(OrigDataPath, saveMatPath, chans=chans)

    # check if the filtered EEG .npz file exists or filter it.
    FilteredPath = os.path.join(saveNpzPath, '{}Hz_{}Hz_{}'.format(bandFiltCutF[0], bandFiltCutF[1], filterType))
    if not os.path.exists(FilteredPath):
        os.makedirs(FilteredPath)

        # load the EEG data from the .mat file one subject by one subject
        files = os.listdir(saveMatPath)
        files = sorted(files)
        for filename in files:
            data_orig = loadmat(os.path.join(saveMatPath, filename), verify_compressed_data_integrity=False)
            x = np.concatenate((data_orig['session1']['x'][0, 0], data_orig['session2']['x'][0, 0]), axis=0)
            y_data = np.squeeze(np.concatenate((data_orig['session1']['y'][0, 0], data_orig['session2']['y'][0, 0]), axis=1))
            fs = data_orig['session1']['s'][0, 0].squeeze().item()

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
            filename = os.path.join(FilteredPath, 's{}.npz'.format(data_orig['SubjectNo'][0, 0]))
            np.savez(filename, x_data=x_data, y_data=y_data, fs=fs)




if __name__ == '__main__':

    chans = None
    # chans = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
    downdampleFactor = 4
    bandFiltCutF = [0.3, 40]
    filterType = 'cheby2_sos'

    # filter the raw data and store the filtered EEG as array into a .np file for each subject
    OrigDataPath = r'/path/to/Original_data/KoreaU_MI_dataset/BCI_dataset/DB_mat'
    saveMatPath = r'/path/to/Original_data/KoreaU_MI_dataset/One_file_one_sub'
    saveNpzPath = r'/path/to/filtered_data/KoreaU_MI'


    preprocessing_ku_dataset(OrigDataPath, saveMatPath, saveNpzPath, chans, downdampleFactor, bandFiltCutF, filterType)

