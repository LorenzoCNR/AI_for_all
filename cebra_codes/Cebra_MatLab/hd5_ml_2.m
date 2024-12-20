

%cd('/media/zlollo/STRILA/CNR neuroscience/cebra_codes/Cebra_for_all/cebra_codes')
% db name
f_name ='manif_data.hdf5';
% gr_name
gr_name='/Cebra_behav';

% db_info
info = h5info(f_name, gr_name);

% structure of the hd5 db
h5disp(f_name);


% 
% labels = h5read(f_name, [gr_name '/labels'])';
% disp('Labels:');
% disp(labels);

%% put data in a matalb structure
%% labels
manif_db=struct();
labels_ = h5read(f_name, [gr_name '/labels']);
manif_db.labels = labels_;  

% all manifolds
for i = 2:length(info.Datasets)
    db_name = info.Datasets(i).Name;
    if startsWith(db_name, 'manif_')
        db_path = [gr_name '/' db_name];
        manif_data = h5read(f_name, db_path);
        disp(['Dataset ' db_name ':']);
        %disp(manif_data);
        manif_db.(db_name) = manif_data;
    end
end

%% extract last data
%labels=manif_db.labels';
% data_1=manif_db.manif_20240405_222955';
%data_1=manif_db.(mnames{end})';
%plot_manif4(manif_db.(mnames{end})', manif_db.labels');

% prepare data in TrialBox format
mnames                      = fieldnames(manif_db);
behavior                    = manif_db.labels;          % nFeatures x nTimes
trialNames                  = {'Left','Right'};         % label names
repTrialType                = behavior(2,:)+1;          % left 1, right 2;
cebra                       = manif_db.(mnames{end});   % nChannels x nTimes (last data in database)

data_trials(1).behavior     = behavior;
data_trials(1).gradients    = data_trials(1).behavior(1,:);
data_trials(1).repTrialType = repTrialType;
data_trials(1).cebra        = cebra; 
data_trials(1).repTrialName = trialNames(data_trials(1).repTrialType);

% plot_scatterGradient
par.plot_scatterGradient                 = plot_scatterGradientParams();
par.plot_scatterGradient.InField         = 'cebra';
par.plot_scatterGradient.InGradient      = 'gradients';
%par.plot_scatterGradient.lats            = [1,3];              % directions to be plot     
par.plot_scatterGradient.lats            = [2,3,1];              % directions to be plot     
par.plot_scatterGradient.reverse         = [false,false,true];   % reverse directions axis
% start gradient color for each class
par.plot_scatterGradient.cmapslight      = [[1.0,0.0,1.0]; ...   % left  start from magenta
                                            [1.0,1.0,0.4]];      % right start from yellow
% end gradient color for each class
par.plot_scatterGradient.cmaps           = [[0.0,1.0,1.0]; ... % left  end in cyan
                                            [0.0,0.5,0.4]];    % right end in green
par.plot_scatterGradinnt.label           = 'cm';
hfg.plot_scatterGradient    = plot_scatterGradient(data_trials,par.plot_scatterGradient);
title(mnames{end},'interpreter','none')