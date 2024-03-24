function data_struct = func0(data_neur,data_behav)

%data_neur=rat_neural
%data_behav=rat_behav

%%% identifico numero trial
data_behav(:,4)=1;
k=1
for i=2:length(data_behav)
    if data_behav(i-1,2)==0 && data_behav(i,2)==1;
        data_behav(i,4)=k+1;
        k=k+1
        
    end
    
end

%%% carryforward Soluzione 1
% %%% più efficiente meno immediata
% tic
% % Numero totale di righe nella colonna 4
% n = length(data_behav(:,4)); 
% i = 1;
% 
%  % Se il valore corrente è 1, go on
% while i <= n
% 
%     if data_behav(i, 4) == 1
%         i = i + 1;
%         continue;
%     end
% 
%     % Se il valore corrente è diverso da 1, 
%     %tag value and fill subsequent rows
%     currentVal = data_behav(i, 4);
%     startIdx = i;
% 
%     % Procedo fino a quando non trovo un valore diverso dal valore corrente 
% 
%     while i <= n && (data_behav(i, 4) == currentVal || data_behav(i, 4) == 1)
%         i = i + 1;
%     end
% 
% 
%     % fine sequenza o fine vettore valori uguali currentval
%     rat_behav(startIdx:i-1, 4) = currentVal;
% end
% 
% elapsedTime = toc; % Salva il tempo trascorso in 'elapsedTime'
% fprintf('Il tempo trascorso è %f secondi.\n', elapsedTime);
% 

%%% carryforward Soluzione 2 
% più immediata meno efficiente

tic

% trovo e taggo valori behavior diversi da 1 (valore colonna inizializata)
idx=find(data_behav(:,4)~=1);

for i=1:length(idx)-1
    s_t=idx(i);
    e_n=idx(i+1)-1;
    data_behav(s_t:e_n,4)=data_behav(idx(i),4);
    
end


if idx(end) < length(data_behav(:,4))
    % imposto l'intervallo dall'ultimo idx trovato fino alla fine del
    % vettore con l'ultimo valore trovato in idx
    data_behav(idx(end):end,4) = data_behav(idx(end),4);
end

elapsedTime = toc; 
fprintf('Il tempo trascorso è %f secondi.\n', elapsedTime);

    
% k = rat_behav(1,4); % Inizializza k con il valore della prima riga per gestire il caso di partenza
% for i=2:length(rat_behav)-1
%     if rat_behav(i,4) ~= k;
%        k=rat_behav(i,4);
%        rat_behav(i+1,4)=k; % Aggiorna k al nuovo valore
%     else
%       rat_behav(i,4)=k ; % Aggiorna il valore attuale con k se non c'è stato un cambio
%     end
% end
% 


%%%aggiungo colonna trial in 
data_neur(:,end+1)=data_behav(:,end);
%data_neur(:,121)=[]


n_trials=max(data_behav(:,4))

c1='spikes'
c2='labels'
c3='times_sec'

data_struct = struct(c1, cell(1, n_trials), c2, cell(1, n_trials), c3, cell(1, n_trials));

n_ms=25
n_s=n_ms/1000

for i = 1:n_trials
    
    idx_spikes = find(data_neur(:, end) == i);
    idx_labels = find(data_behav(:, end) == i);
        
    % Estrai i dati corrispondenti da ciascuna matrice e assegnali ai campi della struttura
    % Assicurati di escludere l'ultima colonna degli identificativi dai dati assegnati
    data_struct(i).spikes = data_neur(idx_spikes, 1:end-1)';
    data_struct(i).labels = data_behav(idx_labels, 1:end-1)';
    n_sp=size(data_struct(i).spikes,2);
    data_struct(i).times_sec = linspace(0,n_sp*0.025,n_sp);
end
end