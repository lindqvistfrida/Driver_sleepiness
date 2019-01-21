%% Drowsi WP4

%%

fp1_3 = edf2struct('fp1 - 3.edf');

%%

Marker = fp1_3.data8.Marker;
plot(Marker)

%% Time 

index_nonzero = find(Marker);
start_index = index_nonzero(1);

end_index = index_nonzero(end);

time = fp1_3.data8.time;

%After this amount of time the experiment started
start_time = time(start_index);

%After this amount of time the experiment ended
end_time = time(end_index);

%%

ECG = fp1_3.data256.ECG;
ECG_time = fp1_3.data256.time;

ECG_start_index = round(start_time)*256;
ECG_end_index = round(end_time)*256;

ECG_start = ECG_time(ECG_start_index); % stämmer
ECG_end = ECG_time(ECG_end_index); % stämmer

%Extract ECG of interest 
ECG = ECG(ECG_start_index:ECG_end_index);

%%

% New time vector

timeSync = zeros(length(time)+start_index,1);

timeSync(start_index:end-1) = time;




