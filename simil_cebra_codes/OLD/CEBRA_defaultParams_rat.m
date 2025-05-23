function   params=CEBRA_defaultParams_rat()
% function params=CEBRA_defaultParams()



%% model_type: hypothesis (che include anche hybrid 
%% quando setti hybrid = True) 
% discovery,
% shuffle





params = struct('model_type',           'hypothesis',       ...
                'mod_arch',             'offset10-model',   ...
                'distance',             'cosine',           ...
                'conditional',          'time_delta',       ...
                'temperature',          1,                  ...       
                'time_offsets',         10,                 ...
                'max_iter',             1000,               ...
                'max_adapt_iter',       500,                ...
                'batch_size',           256,                ...
                'learning_rate',        3e-4,               ...
                'output_dimension',     3,                  ...
                'verbose',              'True',             ...
                'num_hidden_units',     32,                 ...
                'pad_before_transform', 'True',             ...
                'hybrid',               'false'             ...
                );


%%% Una nota sui modelli che si fanno girare (e in particolare i loro
% iperparametri)
%Questi offrono diversi parametri...tipo
% 1) ARCHITETTURA DELLA RETE (default: offset1-model)
% ['offset10-model', 'offset10-model-mse', 'offset5-model', 
% 'offset1-model-mse']
% con dati sintetici offset1 va bene, offset10 va bene con EEG et calcium
% imaging....offset40 quando ci sono immagini...offset5non si sa...
% 2) TEMPERATURE: Fattore per cui scalare la similarità: 
%   parametro che scala la somiglianza tra le coppie positive e negative.
%   Regolando questo parametro, è possibile influenzare quanto fortemente 
%   il modello dovrebbe considerare le coppie come simili o dissimili. 
%   Valori più alti di questo fattore di scala portano alla creazione di
%   embedding più "affilati" e concentrati. In altre parole, aumentando 
%   il valore di questo parametro, gli embedding risultanti saranno più
%   distinti l'uno dall'altro per le coppie negative e più simili tra 
%   loro per le coppie positive. Di fatto,  questo aiuta a migliorare la 
%   capacità del modello di distinguere tra diversi tipi di dati
% 3) OUTPUT DEMENSION: la dimensione dello spaio di arrivo (embedding)
% 4) MAX ITERATIONS: va da sè
% 5) CONDITIONAL La distribuzione condizionata da utilizzare per campionare 
%   i campioni positivi cioè simili al campione di riferimento. I campioni 
%   di riferimento e  quelli negativi vengono estratti da una prior uniforme. 
%   In particolare, sono supoortate 3 tipi di distribuzione
%   -time: I campioni positivi sono scelti in base al loro momento temporale,
%   con un offset temporale fisso rispetto ai campioni di riferimento.
%   Questo significa che i campioni positivi sono quelli che si verificano 
%   in un momento specifico prima o dopo il campione di riferimento
%   -time delta: Questo approccio considera come il comportamento 
%   (o le caratteristiche dei dati) cambia nel tempo. 
%   I campioni positivi sono scelti considerando la distribuzione empirica
%   del comportamento o delle caratteristiche all'interno di un intervallo
%   di tempo definito (time_offset).
%   -delta: Qui, i campioni positivi sono scelti in base a una distribuzione
%   gaussiana centrata intorno al campione di riferimento, 
%   con una deviazione standard (delta) fissa. Ciò significa che i 
%   campioni positivi saranno quelli che sono "vicini" al campione di
%   riferimento secondo una misura quantitativa definita dal delta.
%   Campioni di riferimento e negativi: Sia i campioni di riferimento 
%   che quelli negativi (dissimili dal campione di riferimento) vengono 
%   scelti da una distribuzione uniforme, il che significa che vengono 
%   selezionati casualmente dall'intero set di dati senza una preferenza 
%   specifica.
% 6) DEVICE: 
% 7) VERBOSE: Fa vedere come evolve il training 
% 8) TIME_OFFSET:  Gli "offsets" sono valori che determinano come i
%   campioni (positivi) vengono selezionati rispetto a un punto di riferimento nel
%   tempo. Questi valori sono cruciali per costruire la distribuzione 
%   empirica, ovvero una rappresentazione basata sui dati effettivi di 
%   come variano le caratteristiche dei campioni nel tempo. L'offset può
%   essere un singolo valore fisso, che significa che tutti i campioni 
%   positivi saranno selezionati con lo stesso intervallo di tempo dal
%   campione di riferimento. Alternativamente, può essere una tupla di
%   valori, da cui il modello campiona uniformemente. Questo permette una
%   maggiore varietà e casualità nella selezione dei campioni positivi, 
%   riflettendo diverse possibili distanze temporali rispetto al campione
%   di riferimento. Time offset ha effetto solo se conditional è
%   settata su "time" o "time_delta"
% 9) HYBRID: se Settata su True, il modello verrà allenato usando funzioni 
%    di perdita che distinguono tra campioni in momenti diversi
%   (time contrastive) e tra campioni che presentano diversi comportamennti
%   o stati. (behavio contrative)
% 10)DISTANCE: funzione di distanza usata nel training per definire i
%    campioni positivi e negativi rispetto ai campioni di riferimento
%    Può essere cosine ed euclidean
%%%

%%%% metto dati ratto in formato Mirco
