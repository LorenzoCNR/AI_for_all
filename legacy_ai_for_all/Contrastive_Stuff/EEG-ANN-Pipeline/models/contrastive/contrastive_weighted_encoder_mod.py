# Import nn from torch to define layers and neural models
import torch.nn as nn
import torch.nn.functional as F  # Import functional to access non-stateful functions like 'normalize'

# Define the model as a subclass of nn.Module to integrate with the PyTorch ecosystem
class EncoderContrastiveWeights(nn.Module):
    # Constructor accepts network layers, a label distance function, label weights, temperature, and whether to train temperature
    def __init__(self, layers: nn.Module, labels_distance, labels_weights, temperature=1.0, train_temperature=False):
        super().__init__()  # Call the superclass constructor to initialize inheritance
        # Model layers (must be an nn.Module)
        self.layers = layers  
         # Function to calculate distances between labels
        self.labels_distance = labels_distance 
        # Convert label weights into a PyTorch parameter to track their history in the computation graph
        self.labels_weights = nn.Parameter(torch.tensor(labels_weights, dtype=torch.float), requires_grad=False)

        # Handle temperature as a trainable parameter only if requested
        self.temperature = nn.Parameter(torch.tensor([temperature]), requires_grad=train_temperature)

    # Define the model's forward pass
    def forward(self, x):
         # Apply model layers to input
        f_x = self.layers(x) 
         # Normalize the output along the feature dimension
        f_x = F.normalize(f_x, p=2, dim=1) 
        return f_x

    # Define how to calculate the loss
    def loss(self, f_x, weights):
        # Calculate similarity between all examples in the batch
        # Use mm for matrix-matrix multiplication and adjust for temperature
        psi = torch.mm(f_x, f_x.t()) / self.temperature.exp()  
        # Create a mask for the diagonal
        mask = torch.eye(len(weights), device=weights.device, dtype=torch.bool)  
         # Set the diagonal to '-inf' to exclude it from later calculations
        psi.masked_fill_(mask, float('-inf')) 
        # Compute the logarithm of the weights to stabilize numerical computation
        weights_log = torch.log(1 + weights) 
        # Add the logarithmic weights to the similarity

        psi_weighted = psi.unsqueeze(2) + weights_log 
        # Calculate alignment and uniformity using log-sum-exp for numerical stability
        total_label_weight = self.labels_weights.sum()
        alignment = -torch.sum(self.labels_weights * torch.sum(torch.logsumexp(psi_weighted, dim=1), dim=0))
        uniformity = total_label_weight * torch.sum(torch.logsumexp(psi, dim=1))

        return alignment, uniformity, alignment + uniformity

    # Define how to process a batch of data
    def process_batch(self, batch, optimizer, is_eval=False):
        x, labels = batch  # Extract data and labels from the batch
        if not is_eval: 
            optimizer.zero_grad()  # Clear gradients if not in evaluation mode

        f_x = self.forward(x)  # Compute data representations
        weights = self.labels_distance.get_weights(labels)  # Get weights based on label distances

        alignment, uniformity, total_loss = self.loss(f_x, weights)  # Compute the loss

        if not is_eval: 
            total_loss.backward()  # Propagate the gradient
            optimizer.step()  # Update parameters

        return {'alignment': alignment.item(), 'uniformity': uniformity.item(), 'total_loss': total_loss.item()}
