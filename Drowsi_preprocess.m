%% Drowsi WP4

%% FP1

fp1_1 = edf2struct('FP1 2.edf');

%% Create new struct

WP4_1_1 = fp1_1;


%% Check time

Marker = fp1_1.data8.Marker;
figure(1)
plot(Marker)

index_nonzero = find(Marker);
start_index = index_nonzero(1);

end_index = index_nonzero(end);

time = fp1_1.data8.time;

%After this amount of time the experiment started
start_time = time(start_index);

%After this amount of time the experiment ended
end_time = time(end_index);

%% New time vector

%TimeSync vector for data1
%timeSync = fp1_1.data1.time-fp1_1.data1.time(round(start_index/8));

%TimeSync vector for data8
timeSync = fp1_1.data8.time-fp1_1.data8.time(start_index);

%TimeSync vector for data 256
%timeSync = fp1_1.data256.time-fp1_1.data256.time(start_index*256/8);

%TimeSync vector for data 512
%timeSync = fp1_1.data512.time-fp1_1.data512.time(round(start_index*512/8));

%% Adding timeSync to struct 

%WP4_1_1.data1.timeSync = timeSync;
%WP4_1_1.data8.timeSync = timeSync;
%WP4_1_1.data256.timeSync = timeSync;
%WP4_1_1.data512.timeSync = timeSync;

%% Adding KSS

KSS = zeros(length(time),1);

%Time point 45 min 
index_45 = find(timeSync == 45*60);

% Set values of KSS for driving
KSS(start_index:start_index+5*8) = 3;
KSS(start_index+5*8+1:index_45) = 4;

% Set values that are 0 to NaN
temp = find(~KSS);
KSS(temp) = NaN;

%% Adding KSS to struct

WP4_1_1.data8.KSS = KSS;

%%KSS_table = readtable('KSS Körning.xls')

%% Save struct

PHYS = WP4_1_1;

%%

save('WP4_1_2.mat', 'PHYS')




