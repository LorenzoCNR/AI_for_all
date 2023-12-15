function plot_hippocampus(ax, embedding, label, gray, idx_order)
    if nargin < 4
        gray = false;
    end
    if nargin < 5
        idx_order = [1, 2, 3];
    end
    
    r_ind = label(:,2) == 1;
    l_ind = label(:,3) == 1;
    
    if ~gray
        r_c = label(r_ind, 1);
        l_c = label(l_ind, 1);
        
        try
            colormap(ax, 'viridis');
        catch
            disp('Colormap "viridis" not found, using "jet" instead.');
            colormap(ax, 'jet');
        end
    else
        r_c = 'gray';
        l_c = 'gray';
    end
    
    idx1 = idx_order(1);
    idx2 = idx_order(2);
    idx3 = idx_order(3);
    
    scatter3(ax, embedding(r_ind,idx1), embedding(r_ind,idx2), embedding(r_ind,idx3), 0.5, r_c);
    hold(ax, 'on');
    scatter3(ax, embedding(l_ind,idx1), embedding(l_ind,idx2), embedding(l_ind,idx3), 0.5, l_c);
    
    % Calcola e imposta i limiti degli assi
    all_data = [embedding(r_ind, idx1:idx3); embedding(l_ind, idx1:idx3)];
    xlim(ax, [min(all_data(:,1)) max(all_data(:,1))]);
    ylim(ax, [min(all_data(:,2)) max(all_data(:,2))]);
    zlim(ax, [min(all_data(:,3)) max(all_data(:,3))]);

    grid(ax, 'off');
    % Rimuovi la linea che nasconde i colori degli assi
    % set(ax, 'XColor', 'none', 'YColor', 'none', 'ZColor', 'none');
end


