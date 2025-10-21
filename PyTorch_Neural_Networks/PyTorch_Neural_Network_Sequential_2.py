# Counting the number parameter
# Deep learning models are famous for having a lot of parameters.
# With more parameters come more computational complexity and longer
# training times, and a deep learning practitioner must how many parametes
# there model has.
# In this excersice, we'll first calculate the numbers of parameters manually
# then verifying the result using .numel() method.
#
# Question
# Manually calculate the number of parameters of the model below.

# import torch
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(9, 4),     # (9+1) * 4 = 40
    nn.Linear(4, 2),     # (4+1) * 2 = 10
    nn.Linear(2, 1)      # (2+1) * 1 = 3
)  # Total parameters = 53

Parameters = sum(p.numel() for p in model.parameters())
print(Parameters)
