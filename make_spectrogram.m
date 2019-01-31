%% Make spectrogram

load ASHMI_EEG.mat
train_X = train.train_X;
train_Y = train.train_Y;

%%
%Pick out one training example of 30 s
train_X_1 = train_X(1,1:30*256);

%%
% Uses 2 second windows with 1 sec overlap

s = spectrogram(train_X_1,2*256,1*256);

%%

figure(1)
spectrogram(train_X_1,2*256,1*256)

