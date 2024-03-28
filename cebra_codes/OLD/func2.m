function transformedData = func2(modelPath, dataPath, params)
    script_out_dir  = params.script_out_dir;
    matlab_out_dir  = params.matlab_out_dir;
    command = sprintf('python transform_data.py "%s" "%s" "%s"', modelPath, ...
    script_out_dir, matlab_out_dir);
    % Esegui lo script Python
    [status, ~] = system(command);

    % Verifica se lo script Python è stato eseguito correttamente
    if status == 0
        % Leggi i dati trasformati salvati dallo script Python
    data_struct = load(fullfile(matlab_out_dir, 'transformed_data.mat')); 
    else
        error('Errore durante l''esecuzione dello script Python.');

    end
end

