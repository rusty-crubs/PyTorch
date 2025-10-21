# Import PyTorch
import torch
import torch.nn as nn

input_tensor = torch.Tensor(
    [[2, 7, 9, 5, 3]]
)

# Reorder the items provided to create a Neural Network with three hidden layers and an output of size 2
# Creating Linear_Layar
model = nn.Sequential(
    nn.Linear(5, 20),
    nn.Linear(20, 14),
    nn.Linear(14, 3),
    nn.Linear(3, 2)
)

output = model(input_tensor)
print(output)
