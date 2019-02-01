%% Make spectrogram

load WP1_EEG.mat
train_X = train.train_X;
train_Y = train.train_Y;

%% Make spectrograms of 30 s epochs with KSS labels

rounds = 1598;
KSS = zeros(rounds*10,1);
s = zeros(rounds*10,257,29);
count = 1;
KSS_count = 1;

for n = 1:rounds
    step = 7679;
    for j = 1:step:76800
        if count == 10*n+1
            continue;
        end
        if j == 1
            train_ex = train_X(n,j:j+step);
            KSS(1:10) = train_Y(KSS_count,:);
        else
            train_ex = train_X(n,j+1:j+1+step);
            KSS((KSS_count-1)*10+1:KSS_count*10) = train_Y(KSS_count,:);
        end
        s(count,:,:) = spectrogram(train_ex,2*256,1*256);
        count = count+1;
    end
    KSS_count = KSS_count + 1; 
end

%%
train.s = s;
train.KSS = KSS;

%%
save('WP1_spec_EEG.mat','train')

%% Add two sets of spectrograms together

load ASHMI_spec_EEG.mat

s_1 = train.s;
KSS_1 = train.KSS;
%%
load WP1_spec_EEG.mat

s_2 = train.s;
KSS_2 = train.KSS;

%%

s = [s_1 ; s_2];
KSS = [KSS_1 ; KSS_2];

train.s = s;
train.KSS = KSS;

%%
save('ASHMI_WP1_spec_EEG_1.mat','train','-v7.3')


