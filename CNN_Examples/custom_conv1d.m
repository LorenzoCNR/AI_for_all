function next_layer = custom_conv1d(input, kernel, bias, stride, padding, activation)
    
    %% Kernel has 3 dimensions: kernel size, input chn, output chn (filters)

    % Check if kernel and bias are provided and if they're consistent
    %
    % N.B. ! bias length must be equal to the # of output channels (kernel 
    % third dimension)
    
    if nargin < 2 || isempty(kernel)
        error('Kernel not provided!!.');
    end
    if nargin < 3 || isempty(bias)
        error('Bias not provided!.');
    end

    % Actually the input is a (or a group of M) time series T*1 (T*M)
    % with T>>M ...so Transpose if rows << columns
    if size(input, 1) < size(input, 2)
        input = input';
    end


    % Get input and kernel dimensions
  
    [input_length, input_channels] = size(input);
    [kernel_size, kernel_channels, n_filters] = size(kernel);

    % Check the #input channel is equal to #kernel channels
    if input_channels ~= kernel_channels
        error('#chn in is different from kernel chn in!!');
    end

    % Check the bias size is equal to #kernel third dimension
    if length(bias) ~= n_filters
        error('Bias dimension not matching #kernel filters.');
    end
    
    % Activation functions
    switch activation
        case 'relu'
            activation_fn = @(x) max(0, x);
        case 'sigmoid'
            activation_fn = @(x) 1 ./ (1 + exp(-x));
        case 'gelu'
            activation_fn = @(x) 0.5 * x .* (1 + erf(x / sqrt(2)));
        otherwise
            activation_fn = @(x) x; % Identity function if not specified
    end

    %   'same' padding 
    if nargin >= 5 && strcmp(padding, 'same')
        pad_size = floor(kernel_size / 2);
        input = padarray(input, [pad_size 0], 'replicate', 'both');
    else
        pad_size=0;
    end
    
    % Output dimension
    output_size = floor((input_length- kernel_size + 2 * pad_size) / stride) + 1;
    
    % Initialization
    next_layer = zeros(output_size, size(kernel, 3));
    
    % Convolution for each filter
    for filter_idx = 1:n_filters
        for j = 1:output_size
            % start and end indexes
            start_idx = (j - 1) * stride + 1;
            end_idx = start_idx + kernel_size - 1;
            
            % dot product as a result of convolution
            segment = input(start_idx:end_idx, :);
            conv_result = sum(segment .* kernel(:, :, filter_idx), 'all');
            
            % Add bias
            conv_result = conv_result + bias(filter_idx);
            
            % Apply activation fn
            next_layer(j, filter_idx) = activation_fn(conv_result);
        end
    end
end
