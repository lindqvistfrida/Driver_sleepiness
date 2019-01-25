%% Proprocess EEG

%% Split into epochs

load fp01.mat

epoch_length_sec = 5*60;
epoch_length_samp = 256*epoch_length_sec;

%% Pick out ECG
start_time=find(~isnan(PHYS.data256.timeSync),1,'first');
end_time=find(~isnan(PHYS.data256.timeSync),1,'last');

%ECG signal of interest
EEG = PHYS.data256.EEG2(start_time:end_time);
        
%% Reshaped into epochs

fit = floor(length(EEG)/epoch_length_samp);
EEG = EEG(1 : epoch_length_samp*fit);

EEG_reshaped = reshape(EEG,[epoch_length_samp, fit]);
%% Plot the EEG
figure(1)
plot(EEG_reshaped(:,5))

%% Make KSS vector

KSS = PHYS.data8.KSS;

% Start of the test
startTest=find(~isnan(PHYS.data8.KSS),1,'first');
% End of the test
endTest = find(~isnan(PHYS.data8.KSS),1,'last');

epoch_KSS = epoch_length_sec*8; %Sampling frequency of KSS is 8

KSS = KSS(startTest:endTest);

fit_KSS = floor(length(KSS)/epoch_KSS);

KSS = KSS(1 : epoch_KSS*fit_KSS);
KSS = round(KSS);
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






