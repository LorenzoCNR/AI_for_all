classdef custom_contrastive_loss < nnet.layer.Layer
    
    methods
        function layer = custom_contrastive_loss(name)
            % Constructor for the layer
            layer.Name = name;
            layer.Description = "Custom contrastive loss";
            % Set layer to be a loss layer
            layer.Type = 'custom loss';
        end

        function loss = forwardLoss(~, Y)
            % Forward pass to calculate loss
            ref_rep = Y(1, :);
            pos_rep = Y(2, :);
            neg_reps = Y(3:end, :);
            
            pos_similarity = exp(psi_sim(ref_rep, pos_rep));
            neg_similarity = exp(psi_sim(ref_rep, neg_reps));
            neg_similarity_sum = log(sum(neg_similarity, 1));
            
            loss = -log(pos_similarity + 1e-9) + neg_similarity_sum;
            loss = mean(loss);
        end
        
        function dLdY = backwardLoss(~, Y, ~)  
            % Backward pass to calculate gradient of the loss function
            % Unpack the structured data
            ref_rep = Y(1, :);
            pos_rep = Y(2, :);
            neg_reps = Y(3:end, :);
            
            % Calculate similarities again
            pos_similarity = exp(psi_sim(ref_rep, pos_rep));
            neg_similarity = exp(psi_sim(ref_rep, neg_reps));
            
            % Calculate the derivatives
            dLdPos = -1 ./ (pos_similarity + 1e-9); % derivative of loss with respect to positive similarity
            dLdNeg = 1 ./ (sum(neg_similarity, 1) + 1e-9); % derivative of loss with respect to negative similarities
            
            % TODO: Compute the full derivative of the loss wrt Y, which involves
            % considering how Y contributes to the positive and negative similarities
            % and potentially using the chain rule to propagate the effects.
            dLdY = ... % Implement the specific gradient calculations here
        end
    end
end

function sim = psi_sim(x, y)
    % Calculate L2 norm of rows
    x_norm = sqrt(sum(x.^2, 2));
    y_norm = sqrt(sum(y.^2, 2));
    
    % Calculate cosine similarity
    sim = (x * y') ./ (x_norm * y_norm');
end
