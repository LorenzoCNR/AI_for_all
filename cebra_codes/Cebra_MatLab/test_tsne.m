%%% CHECK DISCORSO ITERAZIONI anche nel transform
 

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
par.tsneCompute                     = tsneComputeParams();
par.tsneCompute.InField             = signal_name;
%cebra_codes_dir                     = 'F:\CNR neuroscience\cebra_codes';

par.tsneCompute.script_fit         = [method_codes_dir 'wrap_tsne_fit.py'];
par.tsneCompute.script_input_dir   = test_directory;
par.tsneCompute.script_output_dir = test_directory;
par.tsneCompute.n_iter             = 1000;
%par.tsneCompute.output_dimension   = 2;
disp(par.tsneCompute);
[~,out.tsneCompute]                = tsneCompute(data_trials,par.tsneCompute);
% cebraProject
par.tsneProject                    = tsneProjectParams();
par.tsneProject.InField            = signal_name;
par.tsneProject.script_transform   = [method_codes_dir 'wrap_tsne_transform.py'];
par.tsneProject.OutField           = 'tsne';
par.tsneProject.model_file         = out.tsneCompute.model_file;

par.tsneroject.script_input_dir   = test_directory;
par.tsneProject.script_output_dir  = test_directory;
[data_trials,out.tsneProject]      = tsneProject(data_trials,par.tsneProject);

%% some plots
%plot2b_mod(data_trials,par.cebraProject)


