function [model_weghts] = func1(data,params,input_dir,output_dir)


% volendo si può dare come input aggiuntivo il numero di trial che vogliamo
% prendere (opzionale)
% if nargin < 3
% 
%         n_tr= []; % Imposta in2 a uno spazio vuoto
%  end
%load("rat_data.mat")
%% matrici vuote per mettere dati ratto
rat_n=[];
rat_b=[];
% # trials
%input_dir=input_directory
%output_dir=output_directory
%data=rat_data
n_tr=size(data,2)

%%% prendo un po' di dati a caso
for i=1:n_tr
    rat_n=[rat_n; data(i).spikes'];
    rat_b=[rat_b;data(i).labels'];
end

save([mfilename 'params.mat'], 'params');
save([mfilename 'rat_n.mat'], 'rat_n');
save([mfilename 'rat_b.mat'], 'rat_b');

cebra_dir=input_dir;
script_name='wrap_py_rat_mod.py'
script_path = [cebra_dir , script_name];
%script_path;
fprintf('Perform CEBRA fit...'); 
save([mfilename 'params.mat'], 'params');

%% Facciamo girare tutto in python 

command = sprintf('python "%s" "%s" "%s"', script_path, input_dir, output_dir);
disp(command)
[status, cmdout] = system(command)
%fprintf('Elapsed Time %g s\n',toc(t));
%system(command)

model_weghts=load(fullfile(output_dir, 'model_struct.mat'));

end