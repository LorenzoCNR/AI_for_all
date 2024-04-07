function hfig=plot_manif4(X_data,labels)

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
par.fine            = 10;                   % round approximation of gradient colorbar
par.lats            = [2,3,1];              % directions to be plot     
par.ms              = 10;                   % marker size
par.widthsub        = 15;                   % subplot size of the main plot
par.reverse         = [false,false,true];   % reverse directions axis
% start gradient color for each class
par.cmapslight      = [[1.0,0.0,1.0]; ...   % left  start from magenta
                       [1.0,1.0,0.4]];      % right start from yellow
% end gradient color for each class
par.cmaps           = [[0.0,1.0,1.0]; ... % left  end in cyan
                       [0.0,0.5,0.4]];    % right end in green


ms                  = par.ms;
lats                = par.lats;
fine                = par.fine;
widthsub            = par.widthsub;
reverse             = par.reverse;
cmaps               = par.cmaps;
cmapslight          = par.cmapslight;

behavior            = labels';          % nFeatures x nTimes
trialNames          = {'Left','Right'};
trialType           = behavior(2,:)+1;  % left 1, right 2;
cebra               = X_data';          % nChannels x nTimes
data_trials(1).behavior     = behavior;
data_trials(1).gradients    = data_trials(1).behavior(1,:);
data_trials(1).repTrialType = trialType;
data_trials(1).cebra        = cebra; 
data_trials(1).repTrialName = trialNames(data_trials(1).repTrialType);

par.InField                 = 'cebra';
par.InGradient              = 'gradients';
InField                     = par.InField;
InGradient                  = par.InGradient;
X_data                      = [data_trials.(InField)];  
trialType                   = [data_trials.repTrialType];
trialName                   = [data_trials.repTrialName];
gradientInfo                = [data_trials.(InGradient)];

hfig=figure; 
subplot(1,widthsub,1:widthsub-2);
hold on; box on; grid on;
    
% sort input data wrt gradientInfo   
[sortGradients,inds]= sort(gradientInfo);
sortEmb             = X_data(:,inds);
sortName            = trialType(inds);
[classes,idc]       = unique(trialType);
names               = trialName(idc);
nClasses            = length(classes);
uniquePoints        = nan(nClasses,1);
for ic = 1:nClasses
    uniquePoints(ic)=length(unique(sortGradients(sortName==classes(ic))));
end
nColors             = mean(uniquePoints);
ulab                = cell(nClasses,1);
gradientMaps        = cell(nClasses,1);
for ic=1:nClasses    
    sortLlab            = sortName==classes(ic);
    slableft            = sortGradients(sortLlab);
    [~,ulab{ic},lbin]   = histcounts(slableft,nColors);
    colStart            = cmapslight(ic,:);
    colEnd              = cmaps(ic,:);
    gmap                = [linspace(colStart(1),colEnd(1),nColors)' ...
                           linspace(colStart(2),colEnd(2),nColors)' ...
                           linspace(colStart(3),colEnd(3),nColors)'];
    gradientMaps{ic}    = gmap;
    scatter3(sortEmb(lats(1),sortLlab), sortEmb(lats(2),sortLlab), sortEmb(lats(3),sortLlab), ms, gmap(lbin,:), 'filled');
    %% sanity check extremalia -> to be deleted
    ifcheck=false;
    if ifcheck
        lwo                 = find(sortLlab);
        [~,wo]              = min(slableft(:,:));
        plot3(sortEmb(lats(1),lwo(wo)), sortEmb(lats(2),lwo(wo)), sortEmb(lats(3),lwo(wo)),'o','markersize',30);
        text (sortEmb(lats(1),lwo(wo)), sortEmb(lats(2),lwo(wo)), sortEmb(lats(3),lwo(wo)),num2str(slableft(wo)),'fontsize',40)
        [~,wo]              = max(slableft(:,:));
        plot3(sortEmb(lats(1),lwo(wo)), sortEmb(lats(2),lwo(wo)), sortEmb(lats(3),lwo(wo)),'o','markersize',30);
        text (sortEmb(lats(1),lwo(wo)), sortEmb(lats(2),lwo(wo)), sortEmb(lats(3),lwo(wo)),num2str(slableft(wo)),'fontsize',40)    
    end
end
view(3);
xlabel(['x' num2str(lats(1))]);
ylabel(['x' num2str(lats(2))]);
zlabel(['x' num2str(lats(3))]);
if reverse(1)
    set(gca, 'XDir','reverse')
end
if reverse(2)
    set(gca, 'YDir','reverse');
end
if reverse(3)
    set(gca, 'ZDir','reverse');
end
% plot colorbars in different subplots
for ic=1:nClasses
    subplot(1,widthsub,widthsub-nClasses+ic);
    axc = gca;
    colormap(axc,gradientMaps{ic});
    hc  = colorbar;
    % double the weight of color bar
    pos=get(hc,'Position');
    pos(3)=2*pos(3);
    set(hc,'Position',pos)  
    axis off
    ulableft            = ulab{ic};
    uTimes              = size(ulableft,2);
    newTicks            = interp1(linspace(hc.Ticks(1),hc.Ticks(end),uTimes),ulableft, linspace(hc.Ticks(1),hc.Ticks(end),length(hc.Ticks)));
    hc.TickLabels       = round(fine*newTicks)/fine;
    hc.Label.String     = names{ic};
end   
end