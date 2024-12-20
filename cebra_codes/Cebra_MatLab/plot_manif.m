%cd('/media/zlollo/STRILA/CNR neuroscience/cebra_codes')
%cd('F:\CNR neuroscience\cebra_codes')
function plot2b_mod(emb,labels)

    %InField                                     = par.InField;
%     OutField                                    = par.OutField;
%     %xfld                                        = par.xfld; %'time';
% 
% %% matrici vuote per mettere dati ratto
%     emb                 =[];
%     rat_b               =[];
%     % reconstruct trials
%     nTrials     =length(data_trials);
%     for iTrial=1:nTrials
%         emb=[emb; data_trials(iTrial).([OutField])'];
%         rat_b=[rat_b; data_trials(iTrial).labels'];
%     end
        
    
    r = labels(:, 2) == 1;
    l = labels(:, 3) == 1;

   % to the right
    t = tiledlayout(1,1);
    ax1 = axes(t); 
    scatter3(emb(l, 1), emb(l, 2), emb(l, 3), 10, labels(l, 1), 'filled');
    colormap(ax1,'winter')
    view(30,30)
    
    % to the left
    ax2 = axes(t);
    scatter3(emb(r, 1), emb(r, 2), emb(r, 3), 10, labels(r, 1), 'filled');
    colormap(ax2,'spring')
    ax2.Visible = 'off';
    view(30,30)
    
    
    cb1 = colorbar(ax1);
    cb1.Layout.Tile = 'east';
    cb1.Label.String = 'Right';
    
    cb2 = colorbar(ax2);
    cb2.Layout.Tile = 'east';
    cb2.Label.String = 'Left';
    title(t,'Cebra')
end