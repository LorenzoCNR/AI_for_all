function [data_trials] = func2_mod(data_trials, model_path, params)
% par=par.cebraCompute
model_p=model_path;
% execinfo    = params.exec;

% matrici vuote per mettere dati ratto
% data_trials=rat_data;
data_n       =[];

%data_b       =[];

nTrials     =length(data_trials);
trial_size=[]
for iTrial=1:nTrials
    data_n=[data_n; data_trials(iTrial).spikes'];
    %%% creo un oggetto dove salvo le dimensioni dei trial che richiamo dopo
    %%% quando vado a fare la riconversione in trial
    trial_size(iTrial)=size(data_trials(iTrial).spikes,2)
    %data_b=[data_b; data_trials(iTrial).labels'];
end

save('data_n.mat', 'data_n');
%save('data_b.mat', 'data_b');

script_name     = params.script_transform;
script_out_dir  = params.script_out_dir;
matlab_out_dir  = params.matlab_out_dir;
disp(['Script Out Dir: ', script_out_dir]);
disp(['MATLAB Out Dir: ', matlab_out_dir]);

%% run fit in python
command = sprintf('python "%s" "%s" "%s" "%s"', script_name, model_p, 'data_n.mat', matlab_out_dir);
fprintf('%s\n', command);  
[status, cmdout] = system(command)
if status ~= 0
    disp('Troubles!!!:');
    disp(cmdout);
else
    disp('Script Output:');
    disp(cmdout);
end
fprintf('%s\n', cmdout);  

out.model_transform = load(fullfile(matlab_out_dir, 'transf_data.mat'));
transf_data = out.model_transform.transformed_data;

% mi creo due indici (uno all'interno del loop) per tracciare gli inidci di ogni
% trial

cums_kk0 = 1; 

for i = 1:length(data_trials)
    kk = trial_size(i); % dimensione del trial corrente
    cums_kk1 = cums_kk0 + kk - 1; % Calcola l'indice finale per questo trial
    data_trials(i).transf_data = transf_data(cums_kk0:cums_kk1, :)'; 
    cums_kk0 = cums_kk1 + 1; %  indice iniziale per il prossimo trial
end


end

