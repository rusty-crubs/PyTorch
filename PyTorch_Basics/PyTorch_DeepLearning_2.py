# While collecting temperature data, you notice the readings are off by two degrees.
# Add two degrees to temperatures tensor after verifying its shape and data type with
# torch to ensure compatibility with the adjustment tensor.
#
# The torch library and the temperature tensor are loaded.

# Import PyTorch
import torch

temperatures = torch.tensor([[72, 75, 78], [70, 73, 76]])
adjustment = torch.tensor([[2, 2, 2], [2, 2, 2]])

# Add the temperatures and adjustment tensor
corrected_temperatures = temperatures + adjustment

print(corrected_temperatures)
