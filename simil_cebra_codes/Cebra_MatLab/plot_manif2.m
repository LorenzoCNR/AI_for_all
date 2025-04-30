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
    par.fine            = 10;
    par.lats            = [2,3,1];      
    par.ms              = 10;
    par.K               = 15;
    ms                  = par.ms;
    lats                = par.lats;
    fine                = par.fine;
    K                   = par.K;
    nTimes              = size(labels,1);
    rLab                = labels(:, 2) == 1;
    lLab                = labels(:, 3) == 1;

    hfig=figure; 
    sbp1=subplot(1,K,1:K-2);
    hold on; box on; grid on;
    sortLabels          = labels(:,1);
    % sortLabels(rLab,:)  = -sortLabels(rLab,:);
    sortLabels(rLab,:)  = sortLabels(rLab,:);

    
    [sortLabels,inds]   = sort(sortLabels);
    sortRlab            = rLab(inds);
    sortLlab            = lLab(inds);
    sortEmb             = emb(inds,:);

    lableft             = labels(lLab,1);
    ulableft            = unique(lableft);
    labright            = -labels(rLab,1);
    ulabright           = unique(labright);
    nColors             = mean([length(ulableft),length(ulabright)]);
    
    slableft            = sortLabels(sortLlab);
    slabright           = sortLabels(sortRlab);
    [~,ulableft,lbin]   = histcounts(slableft,nColors); 
    [~,ulabright,rbin]  = histcounts(slabright,nColors); 

    lmap                = flipud(cool(nColors));
    rmap                = flipud(summer(nColors));
   
    %% check extremalia -> to be deleted
    ifcheck=false;
    if ifcheck
        lwo                 = find(sortLlab);
        [~,wo]              = min(slableft(:,:));
        plot3(sortEmb(lwo(wo), lats(1)), sortEmb(lwo(wo), lats(2)), sortEmb(lwo(wo), lats(3)),'o','markersize',30);
        text(sortEmb(lwo(wo), lats(1)), sortEmb(lwo(wo), lats(2)), sortEmb(lwo(wo), lats(3)),num2str(slableft(wo,:)),'fontsize',40)
        [~,wo]              = max(slableft(:,:));
        plot3(sortEmb(lwo(wo), lats(1)), sortEmb(lwo(wo), lats(2)), sortEmb(lwo(wo), lats(3)),'o','markersize',30);
        text(sortEmb(lwo(wo), lats(1)), sortEmb(lwo(wo), lats(2)), sortEmb(lwo(wo), lats(3)),num2str(slableft(wo,:)),'fontsize',40)
        
        rwo                 = find(sortRlab);
        [~,wo]              = min(slabright(:,:));
        plot3(sortEmb(rwo(wo), lats(1)), sortEmb(rwo(wo), lats(2)), sortEmb(rwo(wo), lats(3)),'o','markersize',30);
        text (sortEmb(rwo(wo), lats(1)), sortEmb(rwo(wo), lats(2)), sortEmb(rwo(wo), lats(3)),num2str(slabright(wo,:)),'fontsize',40)
        [~,wo]              =       max(slabright(:,:));
        plot3(sortEmb(rwo(wo), lats(1)), sortEmb(rwo(wo), lats(2)), sortEmb(rwo(wo), lats(3)),'o','markersize',30);
        text (sortEmb(rwo(wo), lats(1)), sortEmb(rwo(wo), lats(2)), sortEmb(rwo(wo), lats(3)),num2str(slabright(wo,:)),'fontsize',40) 
    end

    cmaps               = nan(nTimes,3);
    cmaps(sortLlab,:)   = lmap(lbin,:);
    cmaps(sortRlab,:)   = rmap(rbin,:);
    % scatter3(sortEmb(:, lats(1)), sortEmb(:, lats(2)), sortEmb(:, lats(3)), ms, cmaps, 'filled');

    scatter3(sortEmb(sortLlab, lats(1)), sortEmb(sortLlab, lats(2)), sortEmb(sortLlab, lats(3)), ms, lmap(lbin,:), 'filled');
    scatter3(sortEmb(sortRlab, lats(1)), sortEmb(sortRlab, lats(2)), sortEmb(sortRlab, lats(3)), ms, rmap(rbin,:), 'filled');

    % view(30,30)
    
    view(3);

    xlabel(['Lat' num2str(lats(1))]);
    ylabel(['Lat' num2str(lats(2))]);
    zlabel(['Lat' num2str(lats(3))]);
    set(gca, 'ZDir','reverse')

    % right
    sbp2=subplot(1,K,K-1);
    ax1 = gca;
    colormap(ax1,rmap);
    hr                  = colorbar;
    pos=get(hr,'Position');
    pos(3)=2*pos(3);
    set(hr,'Position',pos)
    
    uTimes              = size(ulabright,2);
    newTicks            = interp1(linspace(hr.Ticks(1),hr.Ticks(end),uTimes),ulabright, linspace(hr.Ticks(1),hr.Ticks(end),length(hr.Ticks)));
    hr.TickLabels       = '';%round(fine*newTicks)/fine;
    hr.Label.String     = 'Right';

    % left
    axis off
    sbp3=subplot(1,K,K);
    ax2 = gca;
    colormap(ax2,lmap);
    hl                  = colorbar;
    pos=get(hl,'Position');
    pos(3)=2*pos(3);
    set(hl,'Position',pos)
    
    axis off
    uTimes              = size(ulableft,2);
    newTicks            = interp1(linspace(hl.Ticks(1),hl.Ticks(end),uTimes),ulableft, linspace(hl.Ticks(1),hl.Ticks(end),length(hl.Ticks)));
    hl.TickLabels       = round(fine*newTicks)/fine;
    hl.Label.String     = 'Left';

    % im = zeros(nColors,2,3);   

    % im(:,1,:) = rmap;
    % im(:,2,:) = lmap;
    % sbp2 = subplot(1,2,2);
    % imagesc(im)
    % axis off
    
    return
    sbp1.Position = [0.1 0.1 0.6 0.8];
    sbp2.Position = [0.8 0.1 0.1 0.8];
    sbp3.Position = [0.9 0.1 0.1 0.8];

    
    %
    return
    colormap([rmap;lmap]);
    ulabs               = [ulabright,ulableft];
    
    hr                  = colorbar;
    uTimes              = size(ulabs,2);
    newTicks            = interp1(linspace(hr.Ticks(1),hr.Ticks(end),uTimes),ulabs, linspace(hr.Ticks(1),hr.Ticks(end),length(hr.Ticks)));
    hr.TickLabels       = round(fine*newTicks)/fine;
    hr.Label.String     = '<- Left | Right ->';
    xlabel(['Lat' num2str(lats(1))]);
    ylabel(['Lat' num2str(lats(2))]);
    zlabel(['Lat' num2str(lats(3))]);
    set(gca, 'ZDir','reverse')
end