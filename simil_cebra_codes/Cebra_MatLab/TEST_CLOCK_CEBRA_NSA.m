% function TEST_CLOCK_CEBRA_NSA
% PCA Clock, see
% Pezzulo, G., Donnarumma, F., Ferrari-Toniolo, S., Cisek, P., & Battaglia-Mayer, A. (2022). 
% Shared population-level dynamics in monkey premotor cortex during solo action, joint action and action observation. 
% Progress in Neurobiology, 210, 102214.
clear;
par.irng                        = 10;                  % for reproducibility
binWidth                        = 20;
kernSD                          = 30;
rng(par.irng);
%% Step 0. load data in raw format
nf = '~/DATA/SAPIENZA/PROCESSED/all_v5_translated_SK_SUA_t599_all_frontal.mat';                                 % combined sessions
% nf = '~/DATA/SAPIENZA/PROCESSED/all_v5_translated_SK_SUA_t599_all_frequencies20_frontal.mat';                 % alrealdy smoothed % 
% nf = '~/DATA/SAPIENZA/PROCESSED/all_v5_translated_SK_SUA_t599_all_means_frequencies20_cumulative_frontal';    % with mean trials per conditions as first trials 
data_trials                     = load(nf);
data_trials                     = data_trials.dat;
Treset                          = -199;
nTrials                         = length(data_trials);
for it=1:nTrials
    data_trials(it).trialTypeDir= data_trials(it).trialType;         % T1-T8
    data_trials(it).trialTypeCnd= data_trials(it).trialType2;        % joint | obs | solo 
    data_trials(it).trialType   = (data_trials(it).trialTypeCnd-1)*8+data_trials(it).trialTypeDir;
    data_trials(it).trialName   = SapienzaTypeString(data_trials(it).trialTypeCnd,data_trials(it).trialTypeDir);
    data_trials(it).trialNameAll= data_trials(it).trialName;
    data_trials(it).train       = true;
    data_trials(it).valid       = false;
    data_trials(it).test        = false;
    data_trials(it).timespikes  = (1:size(data_trials(it).spikes,2)) + Treset;
end

%% Step 1. prepare data
signal_name                     = 'spikes';       % data savead are in 'spikes' field
signal_process                  = 'y';            % data processed are in 'y' field
% SmoothWindow (moving window smoothing)
par.SmoothWindow                = SmoothWindowParams;
par.SmoothWindow.InField        = signal_name;
par.SmoothWindow.OutField       = signal_process;
par.SmoothWindow.binWidth       = binWidth;
% removeInactives (0 mean channels removal)
par.removeInactive              = removeInactiveParams;
par.removeInactive.InField      = signal_process;
par.removeInactive.OutField     = signal_process;

% function to be execute
par.exec.funname                = {'SmoothWindow','removeInactive'};
data_trials                     = run_trials(data_trials,par);

%%
% meanData
par.meanData                    = meanDataParams;
par.meanData.trialTypes         = [data_trials.trialType];
par.meanData.InField            = signal_process;
par.meanData.OutField           = signal_process;
% AverageWindow (average window selection with sqrt)
par.AverageWindow               = AverageWindowParams;
par.AverageWindow.useSqrt       = true;
par.AverageWindow.InField       = signal_process;
par.AverageWindow.OutField      = signal_process;
par.AverageWindow.binWidth      = binWidth;
% GaussianSmoother (kernel smoothing)
par.GaussianSmoother            = GaussianSmootherParams;
par.GaussianSmoother.InField    = signal_process;
par.GaussianSmoother.OutField   = signal_process;
par.GaussianSmoother.kernSD     = kernSD;       % standard deviation of Gaussian kernel, in msec
par.GaussianSmoother.stepSize   = binWidth;     % time between 2 consecutive datapoints, in msec

% functions to be execute
par.exec.funname                = {'AverageWindow','GaussianSmoother'};
[data_trials, out2]             = run_trials(data_trials,par);

%% Step 2. perform cebra
par.cebraCompute                    = cebraComputeParams();
par.cebraCompute.InField            = signal_process;
nTimes                              = length(data_trials(1).(['time'  par.cebraCompute.InField])); 
for iTrial=1:nTrials
    data_trials(iTrial).labels      = data_trials(iTrial).trialType*ones(1,nTimes);
end
test_directory                      = '/home/donnarumma/TESTS/CEBRA/SAPIENZA/';
par.cebraCompute.script_fit         = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/wrap_cebra_fit.py';
par.cebraCompute.script_input_dir   = test_directory;
par.cebraCompute.script_output_dir  = test_directory;
par.cebraCompute.max_iter           = 100000;
par.cebraCompute.output_dimension   = 12;
par.cebraCompute.OutField           = 'cebra';
disp(par.cebraCompute);
[data_trials_test,out.cebraCompute] = cebraCompute(data_trials,par.cebraCompute);
cebra_output                        = reshape(out.cebraCompute.cebra_output, par.cebraCompute.output_dimension,nTimes,nTrials);
for iTrial=1:nTrials
    data_trials_test(iTrial).(par.cebraCompute.OutField)          = cebra_output(:,:,iTrial);
    data_trials_test(iTrial).(['time' par.cebraCompute.OutField]) = data_trials.(['time' par.cebraCompute.InField]);
end
%%

par.cebraProject                    = cebraProjectParams();
par.cebraProject.InField            = signal_process;
par.cebraProject.script_transform   = '/home/donnarumma/tools/Cebra_for_all/cebra_codes/wrap_cebra_transform.py';
par.cebraProject.OutField           = 'cebra';
par.cebraProject.model_file         = out.cebraCompute.model_file;
par.cebraProject.script_input_dir   = test_directory;
par.cebraProject.script_output_dir  = test_directory;
[data_trials,out.cebraProject]      = cebraProject(data_trials,par.cebraProject);
% Some plots
%% plot_trajectory2D
% meanData
par.meanData                    = meanDataParams;
par.meanData.ifclass            = true;
par.meanData.trialTypes         = [data_trials.trialType];
par.meanData.InField            = par.cebraCompute.OutField;
par.meanData.OutField           = par.cebraCompute.OutField;
par.meanData.P                  = 80;
par.meanData.opt                = [0,0,1];
data_trials_means               = meanData(data_trials, par.meanData);%

cmaps                           = linspecer(24);            % 8 directions x 3 conditions
cmaps                           = cmaps([17:24,1:8,9:16],:);% reset color as desired 
timeField                       = data_trials(1).(['time' par.cebraCompute.OutField]);
[~,i0]                          = min(abs(timeField));      % select   0  ms % TargetOn
istart                          = find(timeField>=-80,1);   % select -80  ms 
iend                            = find(timeField>=250,1);   % select 250  ms
par.plot_trajectory2D           = plot_trajectory2DParams;
par.plot_trajectory2D.keep      = 9:16;                     % which conditions (e.g. obs)
par.plot_trajectory2D.wd        = [1,2];                    % which components
par.plot_trajectory2D.axlab     = 'x';
par.plot_trajectory2D.cmaps     = cmaps;
par.plot_trajectory2D.cmapslight= lightCmaps(par.plot_trajectory2D.cmaps);
par.plot_trajectory2D.explained = [];
par.plot_trajectory2D.InField   = par.cebraCompute.OutField;  % take pca projections to show
par.plot_trajectory2D.istart    = istart;
par.plot_trajectory2D.center    = i0;
par.plot_trajectory2D.iend      = iend;
hfg.plot_trajectory2D           = plot_trajectory2D(data_trials_means, par.plot_trajectory2D);

%% plot pcas dirs on conditions with plot_Each_mean - showing mean and confidence intervals
% remapTypes
par.remapTypes                  = remapTypesParams;
par.remapTypes.selection        = {  1:8  ,  9:16, 17:24 };
par.remapTypes.names            = {'Joint', 'Obs', 'Solo'};
data_trials_conds               = remapTypes(data_trials,par.remapTypes);
% plot_Each_mean
conditions_m                    = unique([data_trials_conds.trialType]);
nConditions_m                   = length(conditions_m);
cmaps                           = linspecer(nConditions_m); % 
cmaps                           = cmaps([2,1,3],:);         % color map for joint, obs and solo

% plot_EachDimVsTime
par.plot_EachDimVsTime              = plot_EachDimVsTimeParams;
par.plot_EachDimVsTime.cmaps        = cmaps;
par.plot_EachDimVsTime.cmapslight   = lightCmaps(par.plot_EachDimVsTime.cmaps);
par.plot_EachDimVsTime.InField      = par.cebraCompute.OutField;  % take pca projection to show;
par.plot_EachDimVsTime.ylabel       = '$pca$';
par.plot_EachDimVsTime.keep         = 1:nConditions_m;
par.plot_EachDimVsTime.nCols        = 4;
par.plot_EachDimVsTime.legplot      = 2;
% GRAPH SUBPLOT TITLES
explained                           = [];%out.pcaCompute.explained;
channels                            = 1:par.cebraCompute.output_dimension;
nChannels                           = length(channels); %% number of graphs
str                                 = cell(nChannels,1);
for ichannel=1:nChannels
    str{ichannel} = sprintf('$$\\tilde{\\mathbf x}_{%d,t}$$',ichannel);
    if ~isempty(explained)
        str{ichannel} = sprintf('%s (%2.1f)',str{ichannel},explained(ichannel));
    end
end
par.plot_EachDimVsTime.titles   = str;

par.meanData                    = meanDataParams;
par.meanData.ifclass            = true;
par.meanData.trialTypes         = [data_trials_conds.trialType];
par.meanData.InField            = par.cebraCompute.OutField;
par.meanData.OutField           = par.cebraCompute.OutField;
par.meanData.P                  = 68;
par.meanData.opt                = [1,0,0];
data_trials_means               = meanData(data_trials_conds, par.meanData);%

hfg.pcas                        = plot_EachDimVsTime(data_trials_means,par.plot_EachDimVsTime);

%% plot pcas dirs on conditions with plot_Each_mean - showing mean and confidence intervals
% remapTypes
par.remapTypes                  = remapTypesParams;
par.remapTypes.selection        = { [ 1, 9,17] , [2,10,18], [3,11,19], [4,12,20], ... 
                                    [ 5,13,21] , [6,14,22], [7,15,23], [8,16,24] };
par.remapTypes.names            = { 'Dir1', 'Dir2', 'Dir3', 'Dir4', ...
                                    'Dir5', 'Dir6', 'Dir7', 'Dir8'};
data_trials_dirs                = remapTypes(data_trials,par.remapTypes);
% plot_Each_mean
conditions_m                    = unique([data_trials_dirs.trialType]);
nConditions_m                   = length(conditions_m);
cmaps                           = linspecer(nConditions_m); % 
%cmaps                           = cmaps([2,1,3],:);         % color map for joint, obs and solo

% plot_EachDimVsTime
par.plot_EachDimVsTime              = plot_EachDimVsTimeParams;
par.plot_EachDimVsTime.cmaps        = cmaps;
par.plot_EachDimVsTime.cmapslight   = lightCmaps(par.plot_EachDimVsTime.cmaps);
par.plot_EachDimVsTime.InField      = par.cebraCompute.OutField;  % take pca projection to show;
par.plot_EachDimVsTime.ylabel       = '$pca$';
par.plot_EachDimVsTime.keep         = 1:nConditions_m;
par.plot_EachDimVsTime.nCols        = 4;
par.plot_EachDimVsTime.legplot      = 2;
% GRAPH SUBPLOT TITLES
explained                           = [];%out.pcaCompute.explained;
channels                            = 1:par.cebraCompute.output_dimension;
nChannels                           = length(channels); %% number of graphs
str                                 = cell(nChannels,1);
for ichannel=1:nChannels
    str{ichannel} = sprintf('$$\\tilde{\\mathbf x}_{%d,t}$$',ichannel);
    if ~isempty(explained)
        str{ichannel} = sprintf('%s (%2.1f)',str{ichannel},explained(ichannel));
    end
end
par.plot_EachDimVsTime.titles   = str;

par.meanData                    = meanDataParams;
par.meanData.ifclass            = true;
par.meanData.trialTypes         = [data_trials_dirs.trialType];
par.meanData.InField            = par.cebraCompute.OutField;
par.meanData.OutField           = par.cebraCompute.OutField;
par.meanData.P                  = 68;
par.meanData.opt                = [1,0,0];
data_trials_means               = meanData(data_trials_dirs, par.meanData);%

hfg.pcas                        = plot_EachDimVsTime(data_trials_means,par.plot_EachDimVsTime);
