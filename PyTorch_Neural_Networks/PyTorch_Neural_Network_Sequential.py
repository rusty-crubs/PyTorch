# Hidden Layer
# model = nn.sequential(
# nn.Linear(n_feature,8), -- n_feature represents number of input features
# nn.Linear(8,4),   -- n_feature , n_classes
# nn.Linear(4, n_classes) -- n_classes represents the number of ouput classes
# )
#
# Input is passed through linear layers
# Layers within nn.sequential() are hidden layers
# n_features and n_classes are defined by the dataset

# # Adding Layers
# model = nn.sequential(
# nn.Linear(10,18), -- 10 weight + 1 bias and 18 neurons, => 11 times 18 is 198 parameters
# nn.Linear(18,20), -- 20 neurons and 18 weight + 1 bias => 19 times 20 is 380 parameters
# nn.Linear(20,5) # Takes 20 and output 5
# )
# Can be calculated using .numel() which returns the number of elements in the tensor
# total = 0
# for parameter in model.parameters():
# total += parameter.numal()
# print(total)
#
# Its time to implement a small neural network containing two linear layers in sequence

# Import PyTorch
import torch
import torch.nn as nn

input_tensor = torch.Tensor([[2, 3, 6, 7, 8, 3, 2, 1]])

# Creating a container for stacking linear layer
model = nn.Sequential(
    nn.Linear(8, 4),
    nn.Linear(4, 1)
)

output = model(input_tensor)
print(output)
