import numpy as np
from sklearn import preprocessing
import argparse
import os

def get_rms(records, multi_channels):
    """
    均方根值 反映的是有效值而不是平均值
    """
    if multi_channels == 1:
        n = records.shape[0]
        rms = 0
        for i in range(n):
            rms_t = np.sum([records[i]**2]) / len(records[i])
            rms = rms + rms_t
        return rms / n

    if multi_channels == 0:
        rms = np.sum([records**2]) / len(records)
        return rms


def get_SNR(signal, noisy):
    snr = 10 * np.log10(signal / noisy)
    return snr

def random_signal(signal, combin_num):
    random_result = []

    for i in range(combin_num):
        random_num = np.random.permutation(signal.shape[0])
        shuffled_dataset = signal[random_num, :]
        shuffled_dataset = shuffled_dataset.reshape(signal.shape[0], signal.shape[1])
        random_result.append(shuffled_dataset)

    random_result = np.array(random_result)

    return random_result


def prepare_data(combin_num, train_per, noise_type):

    file_location = '../data/'  ############ change it to your own location #########
    EEG_all = np.load(file_location + 'EEG_all_epochs.npy')
    noise_all = np.load(file_location + noise_type + '_all_epochs.npy')

    EEG_all_random = np.squeeze(random_signal(signal=EEG_all, combin_num=1))
    noise_all_random = np.squeeze(random_signal(signal=noise_all, combin_num=1))

    if noise_all_random.shape[0] > EEG_all_random.shape[0]:
        reuse_num = noise_all_random.shape[0] - EEG_all_random.shape[0]
        EEG_reuse = EEG_all_random[0: reuse_num, :]
        EEG_all_random = np.vstack([EEG_reuse, EEG_all_random])
    elif noise_all_random.shape[0] < EEG_all_random.shape[0]:
        EEG_all_random = EEG_all_random[0:noise_all_random.shape[0]]

    timepoint = noise_all_random.shape[1]
    train_num = round(train_per * EEG_all_random.shape[0])
    test_num = round(EEG_all_random.shape[0] - train_num)

    train_eeg = EEG_all_random[0: train_num, :]
    test_eeg = EEG_all_random[train_num: train_num + test_num, :]

    train_noise = noise_all_random[0: train_num, :]
    test_noise = noise_all_random[train_num: train_num + test_num, :]

    EEG_train = random_signal(signal=train_eeg, combin_num=combin_num).reshape(combin_num * train_eeg.shape[0],
                                                                               timepoint)
    NOISE_train = random_signal(signal=train_noise, combin_num=combin_num).reshape(combin_num * train_noise.shape[0],
                                                                                   timepoint)

    EEG_test = random_signal(signal=test_eeg, combin_num=combin_num).reshape(combin_num * test_eeg.shape[0],
                                                                             timepoint)
    NOISE_test = random_signal(signal=test_noise, combin_num=combin_num).reshape(combin_num * test_noise.shape[0],
                                                                                 timepoint)

    print(EEG_train.shape)
    print(NOISE_train.shape)

    sn_train = []
    eeg_train = []

    # Adding noise to train
    SNR_train_dB = np.random.uniform(-5, 5, (EEG_train.shape[0]))
    SNR_train = np.sqrt(10 ** (0.1 * (SNR_train_dB)))

    for i in range(EEG_train.shape[0]):

        noise = preprocessing.scale(NOISE_train[i])
        EEG = preprocessing.scale(EEG_train[i])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNR_train[i])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_train.append(signal_noise)
        eeg_train.append(EEG)

    sn_test = []
    eeg_test = []


    SNR_test_dB = np.random.uniform(-5, 5, (EEG_test.shape[0]))
    SNR_test = np.sqrt(10 ** (0.1 * (SNR_test_dB)))

    for j in range(EEG_test.shape[0]):
        noise = preprocessing.scale(NOISE_test[j])
        EEG = preprocessing.scale(EEG_test[j])

        coe = get_rms(EEG, 0) / (get_rms(noise, 0) * SNR_test[j])
        noise = noise * coe
        signal_noise = EEG + noise

        sn_test.append(signal_noise)
        eeg_test.append(EEG)


    X_train = np.array(sn_train)
    y_train = np.array(eeg_train)

    X_test = np.array(sn_test)
    y_test = np.array(eeg_test)

    X_train = np.expand_dims(X_train, axis=1)
    y_train = np.expand_dims(y_train, axis=1)

    X_test = np.expand_dims(X_test, axis=1)
    y_test = np.expand_dims(y_test, axis=1)



    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--n_type', type=str, default='EOG', help='noise version')
    # args = parser.parse_args()
    # print(args)
    for n_type in ['EOG', 'ECG', 'EMG', 'EOG+EMG+ECG']:
        [X_train, y_train, X_test, y_test] = prepare_data(combin_num=11, train_per=0.9, noise_type=n_type)
        data_for_train_test_fold = '../data/data_for_train_test/{}'.format(n_type)
        os.makedirs(data_for_train_test_fold, exist_ok=True)  # 如果文件夹不存在，则创建
        np.save(data_for_train_test_fold+'/X_train.npy'.format(n_type), X_train)
        np.save(data_for_train_test_fold+'/y_train.npy'.format(n_type), y_train)
        np.save(data_for_train_test_fold+'/X_test.npy'.format(n_type), X_test)
        np.save(data_for_train_test_fold+'/y_test.npy'.format(n_type), y_test)