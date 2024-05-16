%%% CHECK DISCORSO ITERAZIONI e parametri in uscita


%cd('/media/zlollo/STRILA/CNR neuroscience/cebra_codes/')
% function TEST_HC11_CEBRA
clear;
% par.irng                            = 10;                  % for reproducibility
% rng(par.irng);
% where are scripts
method_codes_dir                     = '/media/zlollo/STRILA/CNR neuroscience/cebra_codes/';
%cebra_codes_dir                     = 'F:\CNR neuroscience\cebra_codes\';

% where to save outputs
test_directory                      = '/media/zlollo/STRILA/CNR neuroscience/cebra_codes/';
%test_directory                      = 'F:\CNR neuroscience\cebra_codes';
%% Step 0. load data in raw format
% git: 
% mat
data_folder                         = [method_codes_dir 'mat_rat_data/'];
rat_behav_                          = load([data_folder 'rat_behav.mat']);
rat_behav                           = rat_behav_.my_matrix;
rat_neural_                         = load([data_folder 'rat_neural.mat']);
rat_neur                            = rat_neural_.my_matrix;

%% Step 1. Arrange Trials
data_trials                         = hc11ArrangeTrials(rat_neur,rat_behav);
%% Step 2. perform cebra model
% cebraCompute
signal_name                         = 'spikes';
par.umapCompute                     = umapComputeParams();
par.umapCompute.InField             = signal_name;
%cebra_codes_dir                     = 'F:\CNR neuroscience\cebra_codes';

par.umapCompute.script_fit         = [method_codes_dir 'wrap_umap_fit.py'];
par.umapCompute.script_input_dir   = test_directory;
par.umapCompute.script_output_dir  = test_directory;

%%%% 
%par.umapCompute.n_iter             = 1000;
%par.tsneCompute.output_dimension   = 2;
disp(par.umapCompute);
[~,out.umapCompute]                = umapCompute(data_trials,par.umapCompute);
% cebraProject
par.umapProject                    = umapProjectParams();
par.umapProject.InField            = signal_name;
par.umapProject.script_transform   = [method_codes_dir 'wrap_umap_transform.py'];
par.umapaProject.OutField          = 'umap';
par.umapProject.model_file         = out.umapCompute.model_file;

par.umaproject.script_input_dir    = test_directory;
par.umapProject.script_output_dir  = test_directory;
[data_trials,out.umapProject]      = umapProject(data_trials,par.umapProject);

%% some plots
%plot2b_mod(data_trials,par.cebraProject)


