# import PyTorch
import torch
import torch.nn as nn

torch.manual_seed(42)

# input_tensor = torch.tensor([[1.0, 2.0, 3.0]], dtype=None)
input_tensor = torch.rand([3, 3], dtype=None)
# Creating Linear layer
linear_layer = nn.Linear(in_features=3, out_features=6)

output = linear_layer(input_tensor)

print(f"The Shape of input_tensor: {input_tensor.shape}")
print(f"The output:{output}", f"\nShape:{output.shape}",
      f"\nSize:{output.size}")

# The input tensor row x colume, colume must be equal with the in_feature of
# the linear_layer
print("Sequential output\n")
sequence = nn.Sequential(
    nn.Linear(in_features=3, out_features=6),
    nn.Linear(in_features=6, out_features=8),
    nn.Linear(in_features=8, out_features=6)
)

output = sequence(input_tensor)
print(f"The output: {output},\nand its Shape: {output.shape}")
