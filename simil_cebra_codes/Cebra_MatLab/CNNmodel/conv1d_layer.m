function x_out = conv1d_layer(x_in, weights, bias, pad_size, varargin)
    p = inputParser;
    % di default tutto off (no funzione attivazione, no skip connection
    % no normalizzazione
    addOptional(p, 'activation_fn', @(x) x); 
    addOptional(p, 'use_skip_connection', false); 
    addOptional(p, 'normalize_output', false); 
    parse(p, varargin{:});
    
    %%% "parsa gli argomenti"
    activation_fn = p.Results.activation_fn;
    use_skip_connection = p.Results.use_skip_connection;
    normalize_output = p.Results.normalize_output; 
    
    [num_ch_out, num_ch_in, filter_width] = size(weights);
    input_length = size(x_in, 1);  % Assumo x_in sia T x n
    x_out = zeros(input_length, num_ch_out);  % Output T x n_out

    % Padding = same per mantenere le dimensioni
    x_padded = padarray(x_in, [pad_size, 0], 'replicate', 'both');

    % Convoluzione
    for c = 1:num_ch_out
        for i = 1:input_length
            for r = 1:num_ch_in
                for k = 1:filter_width
                    x_out(i, c) = x_out(i, c) + ...
                                  squeeze(weights(c, r, k)) * ...
                                  x_padded(i - 1 + k, r);
                end
            end
            x_out(i, c) = x_out(i, c) + bias(c);

            % skip connection 
            if use_skip_connection && size(x_in, 2) == num_ch_out
                x_out(i, c) = x_out(i, c) + x_in(i, c);
            end
            
            % activation function
            x_out(i, c) = activation_fn(x_out(i, c));
        end
    end
    %% Normalizzazione 
    % Calcolo la norma L2 per il vettore di output al tempo i
    % Normalizza questo vettore di output
    if normalize_output
    for i = 1:input_length % 
        norm = sqrt(sum(x_out(i, :).^2)); 
        if norm > 0
            x_out(i, :) = x_out(i, :) / norm; 
        end
    end
    end


end