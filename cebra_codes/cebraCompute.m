function [data,out] = cebraCompute(data,params,input_dir,output_dir)

%% matrici vuote per mettere dati ratto
rat_n=[];
rat_b=[];
% # trials
%input_dir=input_directory
%output_dir=output_directory
%data=rat_data
nTrials=length(data);

%%% prendo un po' di dati a caso
for iTrial=1:nTrials
    rat_n=[rat_n; data(iTrial).spikes'];
    rat_b=[rat_b; data(iTrial).labels'];
end

save(['params.mat'], 'params');
save(['rat_n.mat'], 'rat_n');
save(['rat_b.mat'], 'rat_b');

cebra_dir=input_dir;
script_name='wrap_py_rat_mod.py';
script_path = [cebra_dir , script_name];
%script_path;
fprintf('Perform CEBRA fit...'); 

%% run fit in python
command = sprintf('python "%s" "%s" "%s"', script_path, input_dir, output_dir);
disp(command);
[status, cmdout] = system(command);
%fprintf('Elapsed Time %g s\n',toc(t));
%system(command)
disp(cmdout);
out.model_weghts = load(fullfile(output_dir, 'model_struct.mat'));
out.cebra_output = load(fullfile(output_dir, 'cebra_output.mat'));

end