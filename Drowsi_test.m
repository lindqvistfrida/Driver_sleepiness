%% Drowsi WP4

%% FP1

fp1_1 = edf2struct('FP1 2.edf');
% fp1_2 = edf2struct('FP1 2.edf');
% fp1_3 = edf2struct('FP1 3.edf');


%% Create new struct

WP4_1_1 = fp1_1;


%%

Marker = fp1_1.data8.Marker;
figure(1)
plot(Marker)

%% Time 

index_nonzero = find(Marker);
start_index = index_nonzero(1);

end_index = index_nonzero(end);

time = fp1_1.data8.time;

%After this amount of time the experiment started
start_time = time(start_index);

%After this amount of time the experiment ended
end_time = time(end_index);

%% ECG

ECG = fp1_1.data256.ECG;
ECG_time = fp1_1.data256.time;

ECG_start_index = round(start_time)*256;
ECG_end_index = round(end_time)*256;

ECG_start = ECG_time(ECG_start_index); % stämmer
ECG_end = ECG_time(ECG_end_index); % stämmer

%Extract ECG of interest 
ECG = ECG(ECG_start_index:ECG_end_index);

%% EOG 

EOG = fp1_1.data512.EOG_vv;
EOG_time = fp1_1.data512.time;

EOG_start_index = round(start_time)*512;
EOG_end_index = round(end_time)*512;

EOG_start = EOG_time(EOG_start_index); % stämmer
EOG_end = EOG_time(EOG_end_index); % stämmer

%Extract ECG of interest 
EOG = EOG(EOG_start_index:EOG_end_index);


%% New time vector

%TimeSync vector for data1
%timeSync = fp1_1.data1-start_time;

%TimeSync vector for data8
%timeSync = time-start_time;

%TimeSync vector for data 256
%timeSync = fp1_1.data256.time-ECG_start;

%TimeSync vector for data 512
timeSync = fp1_1.data512.time-EOG_start;

%%

length_exp = (end_time - start_time)/60; %in minutes

figure(2)
plot(ECG)

%% Adding timeSync to struct 

%WP4_1_1.data512.timeSync = timeSync;

%% Adding KSS

KSS = zeros(52640,1);

%Time point 45 min 
index_45 = find(timeSync == 45*60);

% Set values of KSS for driving to 4
KSS(start_index:index_45) = 4;

% Set values that are 0 to NaN
temp = find(~KSS);
KSS(temp) = NaN;

%% Adding KSS to struct

%WP4_1_1.data8.KSS = KSS;

%%KSS_table = readtable('KSS Körning.xls')

%% Save struct

%PHYS = WP4_1_1;

%save('WP4_1_1.mat', 'PHYS')




