function hfig=plot_manif2(emb,labels)

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

    hfig=figure; hold on; box on; grid on;
    
    lableft             = labels(l,1);
    ulableft            = unique(lableft);
    labright            = -labels(r,1);
    ulabright           = unique(labright);
    nColors             = mean([length(ulableft),length(ulabright)]);
    [~,ulableft,lbin]   = histcounts(lableft,nColors); 
    [~,ulabright,rbin]  = histcounts(labright,nColors); 

    wmap                = cool(nColors);
    
    cmapleft            = wmap(lbin,:);
    scatter3(emb(l, 1), emb(l, 2), emb(l, 3), 10, cmapleft, 'filled');
    fine=100;
    
    smap                = flipud(summer(nColors));
    cmapright           = smap(rbin,:);
    
    scatter3(emb(r, 1), emb(r, 2), emb(r, 3), 10, cmapright, 'filled');
    
    view(30,30)
    
    colormap([wmap;smap]);
    ulabs           = [ulableft,ulabright];
    hr              = colorbar;
   
    hr.TickLabels   = linspace(round(fine*min(ulabs))/fine,round(fine*max(ulabs))/fine, length(hr.Ticks));
    hr.Label.String = '<- Left | Right ->';
end