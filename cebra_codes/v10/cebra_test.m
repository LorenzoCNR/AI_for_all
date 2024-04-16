% function cebra_test
save_path   ='~/PROJS/exec/';
v10_path    = '/home/donnarumma/OneDrive/tools/Cebra_for_all/cebra_codes/v10/';

%          arrange model encode plot
%options = [ true  true   true  true];  % all python checks
options = [false false  false false];   % only matlab plot
close all;
if options(1)
    % arrange Data
    arrangeCommand  =['python ' v10_path 'arrangeData.py '];
    fprintf('Executing %s\n',arrangeCommand)
    [~,message]=system(arrangeCommand);
    disp(message);
end
if options(2)
    % cebra model
    modelCommand    =['python ' v10_path 'cebraModel.py ' save_path 'cebraModelParams.hd5'];
    fprintf('Executing %s\n',modelCommand)
    [~,message]=system(modelCommand);
    disp(message);
end
if options(3)
    % cebra encode
    encodeCommand   = ['python ' v10_path 'cebraEncode.py ' save_path 'cebraEncodeParams.hd5'];
    [~,message]=system(encodeCommand);   
    disp(message);
end
if options(4)
    plotCommand     = ['python ' v10_path 'plotManifold.py ' save_path 'plotCebraParams.hd5'];
    [~,message]=system(plotCommand); 
    disp(message);
end

%% plot in matlab
behavior_filename   = [save_path 'behavior.hd5'];
manifold_filename   = [save_path 'manifold.hd5'];
group_field         = 'data';
behavior_field      = 'behavior';
manifold_field      = 'manifold';
% nChannels x nTimes
manifold        = h5read(manifold_filename, ['/' group_field '/' manifold_field]);
% nFeatures x nTimes
behavior        = h5read(behavior_filename, ['/' group_field '/' behavior_field]);

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
par.plot_scatterGradient.lats            = [2,3,1];             % directions to be plot     
% par.plot_scatterGradient.lats            = [3,1,2];           % directions to be plot     
par.plot_scatterGradient.reverse         = [false,false,true];  % reverse directions axis
% start gradient color for each class
par.plot_scatterGradient.cmapslight      = [[1.0,0.0,1.0]; ...  % left  start from magenta
                                            [1.0,1.0,0.4]];     % right start from yellow
% end gradient color for each class
par.plot_scatterGradient.cmaps           = [[0.0,1.0,1.0]; ...  % left  end in cyan
                                            [0.0,0.5,0.4]];     % right end in green
par.plot_scatterGradinnt.label           = 'cm';
hfg.plot_scatterGradient    = plot_scatterGradient(data_trials,par.plot_scatterGradient);
title('cebra','interpreter','none')