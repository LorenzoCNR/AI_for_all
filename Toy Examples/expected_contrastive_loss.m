function loss = expected_contrastive_loss(model, ref, pos, negs)
    % Calcola le rappresentazioni apprese per i campioni
    ref_rep = forward(model, ref);
    pos_rep = forward(model, pos);
    neg_reps = forward(model, negs);
    
    % Calcola la similarità positiva e negativa
    pos_similarity = exp(psi_sim(ref_rep, pos_rep));
    neg_similarity = exp(psi_sim(ref_rep, neg_reps));
    neg_similarity_sum = log(sum(exp(neg_similarity), 2));
    
    % Calcola la perdita contrastiva come aspettativa
    loss = -log(diag(pos_similarity) + 1e-9) + neg_similarity_sum;
    loss = mean(loss); % Media della perdita
end
