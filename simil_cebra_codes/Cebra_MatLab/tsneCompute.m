function   [data_trials,out] = tsneCompute(data_trials,params)
% function [data_trials,out] = cebraCompute(data_trials,params)
%params=par.tsneCompute
%params=par.tsneCompute
execinfo            = params.exec;
if ~isempty(execinfo); t=tic; fprintf('Function: %s ',mfilename); end
InField             = params.InField;
%% matrici vuote per mettere dati ratto
rat_n               =[];
%rat_b               =[];
% reconstruct trials
nTrials     =length(data_trials);
for iTrial=1:nTrials
    rat_n=[rat_n; data_trials(iTrial).(InField)'];
   % rat_b=[rat_b; data_trials(iTrial).labels'];
end

script_fit          = params.script_fit;
script_input_dir    = params.script_input_dir;
script_output_dir   = params.script_output_dir;

save([script_input_dir 'params.mat'], 'params');
save([script_input_dir 'rat_n.mat'], 'rat_n');
%save([script_input_dir 'rat_b.mat'], 'rat_b');

%% run fit in python
command             = sprintf('python "%s" "%s" "%s" "%s" "%s"', script_fit, script_input_dir, script_output_dir,'rat_n', 'params');
fprintf('%s ',command);
[~, cmdout]         = system(command);
fprintf('%s',cmdout);
%out.model_weghts    = load(fullfile(script_output_dir, 'tsne_struct.mat'));

%tsne_output        = load(fullfile(script_output_dir, 'tsne_output.mat'));
%out.cebra_output    = tsne_output.tsne_output';
out.model_file      = [script_output_dir, 'tsne_model.pkl'];
if ~isempty(execinfo); out.exectime=toc(t); fprintf('| Time Elapsed: %.2f s\n',out.exectime); end

end
