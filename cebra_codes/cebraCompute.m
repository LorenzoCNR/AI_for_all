function [data_trials,out]   = cebraCompute(data_trials,params)
% function [data_trials,out] = cebraCompute(data_trials,par)
execinfo    = params.exec;
if ~isempty(execinfo); t=tic; fprintf('Function: %s ',mfilename); end

%% matrici vuote per mettere dati ratto
rat_n       =[];
rat_b       =[];
% reconstruct trials
nTrials     =length(data_trials);
for iTrial=1:nTrials
    rat_n=[rat_n; data_trials(iTrial).spikes'];
    rat_b=[rat_b; data_trials(iTrial).labels'];
end

save('params.mat', 'params');
save('rat_n.mat', 'rat_n');
save('rat_b.mat', 'rat_b');

script_name     = params.script_name;
script_out_dir  = params.script_out_dir;
matlab_out_dir  = params.matlab_out_dir;
%% run fit in python
command         = sprintf('python "%s" "%s" "%s"', script_name, script_out_dir, matlab_out_dir);
fprintf('%s ',command);
[~, cmdout]     = system(command);
fprintf('%s',cmdout);
out.model_weghts = load(fullfile(matlab_out_dir, 'model_struct.mat'));

cebra_output = load(fullfile(matlab_out_dir, 'cebra_output.mat'));
out.cebra_output=cebra_output.cebra_output;
if ~isempty(execinfo); out.exectime=toc(t); fprintf('| Time Elapsed: %.2f s\n',out.exectime); end

end