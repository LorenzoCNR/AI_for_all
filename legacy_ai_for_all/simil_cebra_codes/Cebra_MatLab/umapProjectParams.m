function   params=umapProjectParams()
% function params=cebraProjectParams()

params.exec             = true;
params.InField          = 'y';
params.OutField         = 'y';
params.script_transform = 'wrap_umap_transform.py';
params.script_output_dir= './';
params.script_input_dir = './';
params.xfld             = 'time';