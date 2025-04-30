%cd('/media/zlollo/STRILA/CNR neuroscience/cebra_codes/')
% function TEST_HC11_CEBRA
clear;
% par.irng                            = 10;                  % for reproducibility
% rng(par.irng);
% where are scripts
cebra_codes_dir                     = '/media/zlollo/STRILA/CNR neuroscience/cebra_codes/';
%cebra_codes_dir                     = 'F:\CNR neuroscience\cebra_codes\';

% where to save outputs
test_directory                      = '/media/zlollo/STRILA/CNR neuroscience/cebra_codes/';
%test_directory                      = 'F:\CNR neuroscience\cebra_codes';
%% Step 0. load data in raw format
% git: 
% mat
data_folder                         = [cebra_codes_dir 'mat_rat_data/'];
rat_behav_                          = load([data_folder 'rat_behav.mat']);
rat_behav                           = rat_behav_.my_matrix;
rat_neural_                         = load([data_folder 'rat_neural.mat']);
rat_neur                            = rat_neural_.my_matrix;

%% Step 1. Arrange Trials
data_trials                         = hc11ArrangeTrials(rat_neur,rat_behav);
%% Step 2. perform cebra model
% cebraCompute
signal_name                         = 'spikes';
par.cebraCompute                    = cebraComputeParams();
par.cebraCompute.InField            = signal_name;
%cebra_codes_dir                     = 'F:\CNR neuroscience\cebra_codes';

par.cebraCompute.script_fit         = [cebra_codes_dir 'wrap_cebra_fit.py'];
par.cebraCompute.script_input_dir   = test_directory;
par.cebraCompute.script_output_dir  = test_directory;
par.cebraCompute.max_iter           = 10000;
par.cebraCompute.output_dimension   = 3;
disp(par.cebraCompute);
[~,out.cebraCompute]                = cebraCompute(data_trials,par.cebraCompute);
% cebraProject
par.cebraProject                    = cebraProjectParams();
par.cebraProject.InField            = signal_name;
par.cebraProject.script_transform   = [cebra_codes_dir 'wrap_cebra_transform.py'];
par.cebraProject.OutField           = 'cebra';
par.cebraProject.model_file         = out.cebraCompute.model_file;

par.cebraProject.script_input_dir   = test_directory;
par.cebraProject.script_output_dir  = test_directory;
[data_trials,out.cebraProject]      = cebraProject(data_trials,par.cebraProject);

%% some plots
plot2b_mod(data_trials,par.cebraProject)


