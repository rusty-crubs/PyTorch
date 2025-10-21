# imported by applying torch.nn in the files
# input neurons are features
# output neurons are Classes
# linear_layer -> .weight and .bias

# Linear Layer Network

# Question: Neural networks oftern contain many layers, but most of them are
# linear layers. Understanding a single linear layers helps us to grasp how
# they works before adding complexity.
#
# Applying a linear layer to an input tensor and observe the output

# Import PyTorch neural network
import torch
import torch.nn as nn

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])

# Create a Linear Layer
linear_layer = nn.Linear(
    in_features=3,
    out_features=2
)

# Pass input_tensor through linear_layer
output = linear_layer(input_tensor)

print(output)

print("\n Weight: ", linear_layer.weight)

print("\n Bias: ", linear_layer.bias)
