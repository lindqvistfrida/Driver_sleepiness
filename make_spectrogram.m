%% Make spectrogram

load WP11_1_1.mat

ECG = PHYS.data256.ECG;

epoch = 2.5*60*256;

figure(1)
spectrogram(ECG(5000:5000+epoch))

figure(2)
spectrogram(ECG(20000:20000+epoch))
