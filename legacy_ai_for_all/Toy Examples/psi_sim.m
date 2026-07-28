function sim = psi_sim(x, y)
    % Calcola la norma L2 delle righe
    x_norm = sqrt(sum(x.^2, 2));
    y_norm = sqrt(sum(y.^2, 2));
    
    % Calcola la similarità del coseno
    sim = (x * y') ./ (x_norm * y_norm');
end