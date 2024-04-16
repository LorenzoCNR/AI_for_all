% function hd5_ml_7
if 0
    % cebra model
    close all;
    v1ph= '/home/donnarumma/OneDrive/tools/Cebra_for_all/cebra_codes/v1/';
    save_path='';
    [~,message]=system(['python ' v1ph 'cebraModel_hd5.py ' v1ph 'config_cebraModel.yaml']);
    disp(message);
    [~,message]=system(['python ' v1ph 'wrap_cebraModel.py ' save_path 'Achilles.hdf5']);
    disp(message);
    % cebra encode
    [~,message]=system(['python ' v1ph 'cebraEncode_hd5.py ' v1ph 'config_cebraEncode.yaml']);   
    disp(message);
    [~,message]=system(['python ' v1ph 'wrap_cebraEncode.py ' save_path 'Achilles_transform.hdf5']); 
    disp(message);
end
%%
save_path='~/tools/Cebra_for_all/cebra_codes/v8/';

behavior_filename   = [save_path 'behavior.hdf5'];
manifold_filename   = [save_path 'manifold.hdf5'];
group_field         = 'data';
behavior_field      = 'behavior';
manifold_field      = 'manifold';
% nChannels x nTimes
manifold        = h5read(manifold_filename, ['/' group_field '/' manifold_field]);
% nFeatures x nTimes
behavior        = h5read(behavior_filename, ['/' group_field '/' behavior_field]);
%%
% dataset_filename= [save_path 'Achilles.hdf5'];
% dataset_group   = '/Achilles_data';
% dataset_info    = h5info(dataset_filename, dataset_group);
% behavior        = h5read(dataset_filename, [dataset_group '/behavior']);

% prepare data in TrialBox format
trialNames                  = {'Left','Right'};         % label names
repTrialType                = behavior(2,:)+1;          % left 1, right 2;

data_trials(1).behavior     = behavior;
data_trials(1).gradients    = data_trials(1).behavior(1,:);
data_trials(1).repTrialType = repTrialType;
data_trials(1).cebra        = manifold; 
data_trials(1).repTrialName = trialNames(data_trials(1).repTrialType);

% plot_scatterGradient
par.plot_scatterGradient                 = plot_scatterGradientParams();
par.plot_scatterGradient.InField         = 'cebra';
par.plot_scatterGradient.InGradient      = 'gradients';
%par.plot_scatterGradient.lats            = [1,3];              % directions to be plot     
par.plot_scatterGradient.lats            = [2,3,1];              % directions to be plot     
% par.plot_scatterGradient.lats            = [3,1,2];              % directions to be plot     
par.plot_scatterGradient.reverse         = [false,false,true];   % reverse directions axis
% start gradient color for each class
par.plot_scatterGradient.cmapslight      = [[1.0,0.0,1.0]; ...   % left  start from magenta
                                            [1.0,1.0,0.4]];      % right start from yellow
% end gradient color for each class
par.plot_scatterGradient.cmaps           = [[0.0,1.0,1.0]; ... % left  end in cyan
                                            [0.0,0.5,0.4]];    % right end in green
par.plot_scatterGradinnt.label           = 'cm';
hfg.plot_scatterGradient    = plot_scatterGradient(data_trials,par.plot_scatterGradient);
title('cebra','interpreter','none')