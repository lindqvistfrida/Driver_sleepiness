%% Proprocess EEG

%% Split into epochs

load 13_3.mat

epoch_length_sec = 2.5*60;
epoch_length_samp = 256*epoch_length_sec;

%% Pick out ECG

% Start of the test at 8Hz
start_time_8 = find(PHYS.data8.timeSync>=CAN.timeSync(find(CAN.timeSync == 0,1,'last')),1,'first');
% End of the test at 8 Hz
end_time_8 = find(~isnan(PHYS.data8.KSS),1,'last');

% Start of the test at 256 Hz
start_time = find(PHYS.data256.timeSync >= PHYS.data8.timeSync(start_time_8),1,'first');
% End of the test at 256 Hz
end_time = find(PHYS.data256.timeSync >= PHYS.data8.timeSync(end_time_8),1,'first');

%ECG signal of interest
EEG = PHYS.data256.EEG2(start_time:end_time);
        
%% Reshaped into epochs

fit = floor(length(EEG)/epoch_length_samp);
EEG = EEG(1 : epoch_length_samp*fit);

EEG_reshaped = reshape(EEG,[epoch_length_samp, fit]);

%% Make KSS vector

KSS = PHYS.data8.KSS;

% Start of the test
startTest = find(PHYS.data8.timeSync>=CAN.timeSync(find(CAN.timeSync == 0,1,'last')),1,'first');
% End of the test
endTest = find(~isnan(PHYS.data8.KSS),1,'last');

epoch_KSS = epoch_length_sec*8; %Sampling frequency of KSS is 8

KSS = KSS(startTest:endTest);

fit_KSS = floor(length(KSS)/epoch_KSS);

KSS = KSS(1 : epoch_KSS*fit_KSS);

KSS_reshaped = reshape(KSS,[epoch_KSS, fit_KSS]);
KSS_labels = KSS_reshaped(1,:);

%% Make data labels binary

for i = 1:length(KSS_labels)
    if KSS_labels(i) == 7
        KSS_labels(i) = 0;
    else if KSS_labels(i) > 7
         KSS_labels(i) = -1;
        else KSS_labels(i) = 1;
        end
    end
end

zero_index = find(~KSS_labels);

KSS_labels(zero_index) = [];
EEG_reshaped(:,zero_index) = [];






