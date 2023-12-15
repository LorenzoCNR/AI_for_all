function plot2c(cebra_struct, labels)
    
    fig = figure;
    nnames = fieldnames(cebra_struct);
    models = {'Pos Only', 'Dir Only', 'Pos+Dir', 'Pos SHuffled'...
        'Dir Shuffled', 'Pos + Dir Shuffled'};

    % Ciclo per ogni modello
    for i = 1:numel(models)
        ax = subplot(2, 3, i, 'projection', 'perspective', 'Parent', fig);
        hold on;

        %  embedding e labels per modello corrente
        emb = cebra_struct.(nnames{i});
        % Controllo del numero di argomenti per 'idx_order'
        if nargin < 4
            idx_order = [1, 2, 3];
        end
    
        % Estraggo indici per i gruppi di punti
        r_ind = labels(:,2) == 1;
        l_ind = labels(:,3) == 1;
    
        % Selezione delle colonne per x, y, z in base a 'idx_order'
        idx1 = idx_order(1);
        idx2 = idx_order(2);
        idx3 = idx_order(3);
    
        % Plot dei punti per ciascun gruppo 
        scatter3(ax, emb(r_ind, idx1), emb(r_ind, idx2), emb(r_ind, idx3), 0.5, 'filled');
        %hold(ax, 'on');
        scatter3(ax, emb(l_ind, idx1), emb(l_ind, idx2), emb(l_ind, idx3), 0.5, 'filled');
          
        

        % Impostazioni degli assi
        title(models{i});
        view(30, 30);
        axis off;
    end


end


% 
