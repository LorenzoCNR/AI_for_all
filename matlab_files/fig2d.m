function fig2d(data, start_idx, framerate, llength, linewidth)

   
    fig = figure;
    ax1_traj = axes('Parent', fig);

   

    % Plot della Traiettoria Reale
    plot(ax1_traj, framerate * (0:2:llength-2), ...
        data.label_test(start_idx:2:start_idx+llength-2, 1) * 100, ...
        'k-', 'LineWidth', linewidth, 'DisplayName', 'Ground Truth');

    hold(ax1_traj, 'on');

    % Plot della Traiettoria Predetta
    plot(ax1_traj, framerate * (0:2:llength-2), ...
        data.pred_posdir_decode(start_idx:2:start_idx+llength-2, 1) * 100, ...
        '-', 'Color','green', 'LineWidth', linewidth, 'DisplayName', 'CEBRA-Behavior');

    % Plot della Traiettoria Shuffle
    plot(ax1_traj, framerate * (0:10:llength-2), ...
        data.pred_posdir_shuffled_decode(start_idx:10:start_idx+llength-2, 1) * 100, ...
        '--', 'Color', 'red', 'LineWidth', linewidth, 'DisplayName', 'CEBRA-Shuffle');

   % assi
    set(ax1_traj, 'YTick', linspace(0, 160, 5), 'XLim', [-1, 17.5], 'XTick', linspace(0, 17.5, 8));
    %ax1_traj.XAxis.Bounds = [0, 17.5];
    %ax1_traj.YAxis.Bounds = [0, 160];
    xlabel(ax1_traj, 'Time [s]');
    ylabel(ax1_traj, 'Position [cm]');

    % legenda
    legend(ax1_traj, 'Location', 'northoutside', 'Orientation', 'horizontal', 'Box', 'off', 'FontSize', 8);

    % on
    set(ax1_traj, 'Box', 'off');
end