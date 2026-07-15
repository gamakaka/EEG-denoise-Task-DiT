# 该模块用于生成混合噪声
from Data_Preparation import data_prepare_eegdnet as dp
import numpy as np
 

def regenerate_noise(EEG, Noise):
    if EEG.shape[0] < Noise.shape[0]:
        Noise = Noise[0:EEG.shape[0], :]
    else:
        reuse_num = EEG.shape[0] - Noise.shape[0]
        noise_reuse = Noise[0 : reuse_num, :]
        Noise = np.vstack([noise_reuse, Noise])


    SNR_dB = np.random.uniform(-5.0, 5.0, (EEG.shape[0]))
    print(EEG.shape, Noise.shape,SNR_dB.shape)
    SNR = 10 ** (0.05 * (SNR_dB))
    #SNR_train=signal/noise

    # generate noise with random SNR

    NOISE_adjust=[]
    for i in range (EEG.shape[0]):
        eeg=EEG[i].reshape(EEG.shape[1])
        noise=Noise[i].reshape(Noise.shape[1])

        coe=dp.get_rms(eeg, 0)/(dp.get_rms(noise, 0)*SNR[i])
        noise = noise*coe

        NOISE_adjust.append(noise)
    return np.array(NOISE_adjust)

if __name__ == '__main__':
    # 生成的混合噪声类型
    generate_noise_type = 'EOG+EMG+ECG'
    raw_data_ubuntu_path = 'datas/EEGdenoiseNet-master-data/'

    EEG_all = np.load(raw_data_ubuntu_path + 'EEG_all_epochs.npy')
    Noise_EOG_all = np.load(raw_data_ubuntu_path + 'EOG_all_epochs.npy')
    Noise_EMG_all = np.load(raw_data_ubuntu_path + 'EMG_all_epochs.npy')
    Noise_ECG_all = np.load(raw_data_ubuntu_path + 'ECG_all_epochs.npy')

    # 调用函数，根据随机的信噪比重新生成噪声
    new_eog = regenerate_noise(EEG_all, Noise_EOG_all)
    new_emg = regenerate_noise(EEG_all, Noise_EMG_all)
    new_ecg = regenerate_noise(EEG_all, Noise_ECG_all)
    # 将新生成的噪声相加得到混合噪声
    combine_noise = new_eog + new_emg +new_ecg

    data_processed_ubuntu_path = 'datas/EEGdenoiseNet-master-data/'
    np.save(data_processed_ubuntu_path + generate_noise_type + '_all_epochs.npy', combine_noise)


