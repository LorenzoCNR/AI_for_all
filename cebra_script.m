main_folder='set your folder in'
cd(main_folder)

%path_to="F:\CNR neuroscience\Effettivo\Working On\Learning Latent Variable"
%pyenv('ExecutionMode', 'OutOfProcess')


%setenv('LD_LIBRARY_PATH', '/home/zlollo/anaconda3/envs/cebra/lib')
%pyenv('Version', '/home/zlollo/anaconda3/envs/cebra0/bin/python');
%% UBUNTU (shell digitare 'which python')

environment

pyenv('Version', '/home/zlollo/anaconda3/envs/cebra/bin/python', ...
    'ExecutionMode', 'InProcess');

    

py.sys.path
% unwanted_paths = {'', '/home/zlollo', '/home/zlollo/CNR/lfads', '/home/zlollo/CNR/lfads-run-manager/src'};
% %updated_py_path = {};
% % Iterate through the unwanted paths
% for i = 1:length(unwanted_paths)
%     % While the unwanted path is in current_py_path, remove it
%     while any(strcmp(current_py_path, unwanted_paths{i}))
%         py.sys.path.remove(unwanted_paths{i});
%         % Update current_py_path after each removal
%         current_py_path = cell(py.sys.path);
%     end
%end
 %% WINDOWS (anaconda prompt digitare 'where python')
 %pyenv('Version', 'C:\Users\loren\anaconda3\envs\cebra\python.exe', ...
 %    'ExecutionMode', 'InProcess');
%% prova a metterlo con join dir
%%% TEST

py.importlib.import_module('os');
py.importlib.import_module('sys');
py.print('Hello from Python');
jl = py.importlib.import_module('joblib');
pytorch = py.importlib.import_module('torch');
[status, cmdout] = system('python -c "import torch; print(torch.__version__)"');
if status == 0
    disp(['PyTorch Version: ', cmdout]);
else
    disp('Error executing Python command');
end
np = py.importlib.import_module('numpy');
pd = py.importlib.import_module('pandas');
%plt = py.importlib.import_module('matplotlib.pyplot');
sklearnMetrics = py.importlib.import_module('sklearn.metrics');

Cebra = py.importlib.import_module('cebra');
cebraDatasets = py.importlib.import_module('cebra.datasets');
%py.importlib.import_module('h5py');

%%%% qui aggiungere la path dove mettiamo i dati

d_folder='/home/zlollo/CNR/Cebra_for_all/datasets/'

data_folder=addpath(genpath(d_folder));

fileName='rat_hippocampus/achilles.jl'
filePath = which(fileName);
filePath
% Check if the file was found
if isempty(filePath)
    disp(['File ' fileName ' not found on the MATLAB path.']);
else
    disp(['File found: ' filePath]);
end
data = jl.load(filePath);

neuralData = data{'spikes'};

behaviorData = data{'position'};
%%% imposto le dimensioni corrette
neuralDataMat = double(py.array.array('d', py.numpy.nditer(neuralData)));
behaviorDataMat = double(py.array.array('d', py.numpy.nditer(behaviorData)));
neuralDataMat = reshape(neuralDataMat, [120, 10178]).';
behaviorDataMat = reshape(behaviorDataMat, [3, 10178]).';

%%% check dimensions
[dataRows, dataCols] = size(behaviorDataMat);
disp(['behaviorDataMat dimensions: ', num2str(dataRows), ' rows, ', num2str(dataCols), ' cols']);

%%%% PLOT
figure('Units', 'inches', 'Position', [0 0 12 4]);

% Plot neural data
subplot(1, 2, 1);
imagesc(neuralDataMat(1:5000, :)');
colormap('gray');
ylabel('Neuron #');
xlabel('Time [s]');
xticks(linspace(0, 5000, 5));
xticklabels(linspace(0, 0.025 * 5000, 5));

% Plot position data
subplot(1, 2, 2);
scatter(1:5000, behaviorDataMat(1:5000, 1), 1, 'filled', 'MarkerFaceColor', [0.5 0.5 0.5]);
ylabel('Position [m]');
xlabel('Time [s]');
xticks(linspace(0, 5000, 5));
xticklabels(linspace(0, 0.025 * 5000, 5));

drawnow;

%%% create now csv files to store data for python
csvwrite('neuralData.csv', neuralDataMat);
csvwrite('behaviorData.csv', behaviorDataMat);
%%% o anche
save('data.mat', 'neuralDataMat', 'behaviorDataMat');
load('data.mat')

%%%% faccio girare i primi modelli cebra in Python 
% Cebra- behaviour  addestrea un modello con output 3d che usa info 
% posiionali (posiz e direz)con l'uso di una variabili ausiliaria, 
% time_delta, il tempo durante il trainingdel modello Il modello 
% Cebra-shuffled, viene usato come coontrollo Il modello Cebra-time 
% utilizza solo info  temporali  e non comportamentali (posizionali)
% Cebra hybrid infine utilizza una combinazione di informazioni temporali e
%  comportamentali in modo più integrato e bilanciato.
% Invece di trattare le informazioni temporali come un semplice contesto 
% ausiliario, "CEBRA-Hybrid" fonde queste informazioni con le variabili 
% comportamentali (posizione e direzione) per costruire un modello più 
% complesso  che tiene conto sia del comportamento che del tempo 
% in maniera paritaria.


%%% li salvo in un mat cebra_1step_output

cebra_1step_output=load("cebra_1step_output.mat")

cebra_posdir3 = cebra_1step_output.cebra_posdir3;
cebra_posdir_shuffled3 = cebra_1step_output.cebra_posdir_shuffled3;
cebra_time3 = cebra_1step_output.cebra_time3;
cebra_hybrid = cebra_1step_output.cebra_hybrid;
behavior_data=behaviorDataMat





%%% viisualizziamo gli spazi embedded generati con Cebra



% Crea le figure
figure;
ax1 = subplot(1, 4, 1, 'projection', 'perspective');
plot_hippocampus(ax1, cebra_posdir3, behavior_data);

ax2 = subplot(1, 4, 2, 'projection', 'perspective');
plot_hippocampus(ax2, cebra_posdir_shuffled3, behavior_data);

ax3 = subplot(1, 4, 3, 'projection', 'perspective');
plot_hippocampus(ax3, cebra_time3, behavior_data);

ax4 = subplot(1, 4, 4, 'projection', 'perspective');
plot_hippocampus(ax4, cebra_hybrid, behavior_data);

% Aggiungi titoli ai subplot, se necessario
title(ax1, 'CEBRA-Behavior');
title(ax2, 'CEBRA-Shuffled');
title(ax3, 'CEBRA-Time');
title(ax4, 'CEBRA-Hybrid');

%%&%%%% TEST IPOTESI:ADDESTRIAMO MODELLI CON DIVERSE IPOTESI %%%%%%
%  sull'encoding posizionale dell'ippocampo. L'obiettivo è confrontare
%  diversi modelli CEBRA-Behavior addestrati utilizzando diverse variabili 
%  comportamentali: solo la posizione, solo la direzione, entrambe queste 
%  variabili, e modelli di controllo con variabili comportamentali mescolate
%  casualmente (shuffled).
% Si utilizza  quindi una dimensione del modello predefinita. 
% Nel lavoro originale descritto nel documento, sono state utilizzate dimensioni 
% del modello che variano da 3 a 64 per analizzare i dati dell'ippocampo, e si è 
% osservata una topologia coerente attraverso queste diverse dimensioni.
% Per le analisi di decodifica successive, si utilizzerà un set di dati diviso, 
% con l'80% dei dati train e il 20% test. 
% Solo il train set sarà  per addestrare i modelli.
% In sintesi, lo scopo di questo esperimento è di esplorare come differenti tipi
%  di variabili comportamentali (posizione, direzione, entrambe, o mescolate) 
%  influenzino la capacità dei modelli CEBRA-Behavior di codificare informazioni
%  sull'ippocampo. 
% 

cebra_2nd_output_hyp_test=load("cebra_2nd_output_hyp_test")

cebra_pos_all = cebra_2nd_output_hyp_test.cebra_pos_all;
cebra_dir_all = cebra_2nd_output_hyp_test.cebra_dir_all;
cebra_posdir_all = cebra_2nd_output_hyp_test.cebra_posdir_all;
cebra_pos_shuffled_all = cebra_2nd_output_hyp_test.cebra_pos_shuffled_all;
cebra_dir_shuffled_all = cebra_2nd_output_hyp_test.cebra_dir_shuffled_all;
cebra_posdir_shuffled_all = cebra_2nd_output_hyp_test.cebra_posdir_shuffled_all;

%Plot
fig = figure;

% Subplot 1: Position Only
ax1 = subplot(2, 3, 1, 'projection', 'perspective');
plot_hippocampus(ax1,cebra_pos_all, behavior_data, false);
title(ax1, 'Position Only');

% Subplot 2: Direction Only
ax2 = subplot(2, 3, 2, 'projection', 'perspective');
plot_hippocampus(ax2, cebra_dir_all, behavior_data, false);
title(ax2, 'Direction Only');

% Subplot 3: Position + Direction
ax3 = subplot(2, 3, 3, 'projection', 'perspective');
plot_hippocampus(ax3, cebra_posdir_all, behavior_data, false);
title(ax3, 'Position + Direction');

% Subplot 4: Position Shuffled
ax4 = subplot(2, 3, 4, 'projection', 'perspective');
plot_hippocampus(ax4, cebra_pos_shuffled_all, behavior_data, false);
title(ax4, 'Position Shuffled');

% Subplot 5: Direction Shuffled
ax5 = subplot(2, 3, 5, 'projection', 'perspective');
plot_hippocampus(ax5, cebra_dir_shuffled_all, behavior_data, false);
title(ax5, 'Direction Shuffled');

% Subplot 6: Position + Direction Shuffled
ax6 = subplot(2, 3, 6, 'projection', 'perspective');
plot_hippocampus(ax6, cebra_posdir_shuffled_all, behavior_data, false);
title(ax6, 'Position + Direction Shuffled');

% Visualizza la figura
drawnow;


%%%%% PLOTTIAMO LA PERDITA DEI MODELLI 

models_loss=load("model_loss.mat");

loss_pos_dir = models_loss.loss_pos_dir;
loss_pos = models_loss.loss_pos;
loss_dir = models_loss.loss_dir;
loss_pos_dir_shuffle = models_loss.loss_pos_dir_shuffle;
loss_pos_shuffle = models_loss.loss_pos_shuffle;
loss_dir_shuffle =models_loss.loss_dir_shuffle;

% Plot delle perdite dei modelli
plot(ax, loss_pos_dir, 'Color', '#00BFFF', 'DisplayName', 'position+direction'); % deepskyblue
hold(ax, 'on'); % Mantiene il grafico corrente per sovrapporre le altre linee
plot(ax, loss_pos, 'Color', [0 191/255 1 0.3], 'DisplayName', 'position'); % Colore più chiaro
plot(ax, loss_dir, 'Color', [0 191/255 1 0.6], 'DisplayName', 'direction'); % Colore più chiaro
plot(ax, loss_pos_dir_shuffle, 'Color', '#808080', 'DisplayName', 'pos+dir, shuffled'); % grigio
plot(ax, loss_pos_shuffle, 'Color', [128/255 128/255 128/255 0.3], 'DisplayName', 'position, shuffled'); % grigio più chiaro
plot(ax, loss_dir_shuffle, 'Color', [128/255 128/255 128/255 0.6], 'DisplayName', 'direction, shuffled'); % grigio più chiaro

legend(ax, 'show');
xlabel(ax, 'Iterations');
ylabel(ax, 'InfoCNE Loss ');
title(ax, 'Model Loss Comparison');

hold(ax, 'off');








