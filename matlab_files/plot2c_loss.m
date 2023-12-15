function plot2c_loss(cebra_struct, labels)
    % Definizione colormap
    cmapL = [linspace(1, 0, 256)', linspace(0, 1, 256)', linspace(1, 1, 256)']; % fucsia a celeste
    cmapR = [linspace(0, 1, 256)', linspace(1, 1, 256)', linspace(0, 1, 256)']; % verde a giallo

   
    % fig = figure;
    % nnames = fieldnames(cebra_struct);
    % models = {'hypothesis', 'shuffled', 'discovery', 'hybrid'};
    % 
    % % Ciclo per ogni modello
    % for i = 1:numel(models)
    %     ax = subplot(1, 4, i, 'Parent', fig);
    %     hold on;
    % 


fig = figure;
ax = axes(fig);
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

        % Estrai gli embedding e le etichette per il modello corrente
        emb = cebra_struct.(nnames{i});

        % Maschere per destra e sinistra
        r = labels(:, 2) == 1;
        l = labels(:, 3) == 1;

        % Calcolo degli indici della colormap
        indsL = min(floor(labels(l, 1) * 255) + 1, 256);
        indsR = min(floor(labels(r, 1) * 255) + 1, 256);

        % Assegna i colori in base all'intensità e alle colormap per sinistra e destra
        clrL = cmapL(indsL, :);
        clrR = cmapR(indsR, :);

        % Plot per sinistra e destra
        scatter3(ax, emb(l, 1), emb(l, 2), emb(l, 3), 10, clrL, 'filled');
        scatter3(ax, emb(r, 1), emb(r, 2), emb(r, 3), 10, clrR, 'filled');

        % Impostazioni degli assi
        title(models{i});
        view(30, 30);
        axis off;
    end
    % Aggiunta delle colorbar fittizie
    % Colorbar per la sinistra
    % Aggiunta delle colorbar fittizie
    % Colorbar per la sinistra
    cbarAxL = axes('Position', [.91 .11 .01 .8150], 'CLim', [0 1], 'Parent', fig);
    colormap(cbarAxL, cmapL);
    cbarL = colorbar(cbarAxL, 'Position', [.91 .11 .01 .8150]);
    cbarL.Label.String = 'label L'; % Imposta l'etichetta per la colorbar sinistra
    cbarAxL.Visible = 'off';

    % Colorbar per la destra
    cbarAxR = axes('Position', [.93 .11 .01 .8150], 'CLim', [0 1], 'Parent', fig);
    colormap(cbarAxR, cmapR);
    cbarR = colorbar(cbarAxR, 'Position', [.93 .11 .01 .8150]);
    cbarR.Label.String = 'label R'; % Imposta l'etichetta per la colorbar destra
    cbarAxR.Visible = 'off';

end
