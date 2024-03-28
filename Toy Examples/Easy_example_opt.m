%%% toy model come esempio 
%% 

% Replace following line with path to python in you env.
pe = pyenv('Version', '/usr/bin/python3');

% check
if pe.Status == "NotLoaded" || pe.Status == "Failure"
    error('Python environment non è stato caricato correttamente.')
end



% Run Python cfi da MATLAB
py_out = py.run(py_file);


% Ptrint python input(if)
if ~isempty(py_out)
    disp(py_out)
end