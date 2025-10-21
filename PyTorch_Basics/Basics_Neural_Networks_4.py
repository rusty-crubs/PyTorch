# Import PyTorch
import torch

# Creating tensor
x = torch.arange(1, 10).reshape(1, 3, 3)
print(f"Target value:{x}")
print(f"Target Shape:{x.shape}")

# Get all value of 0th and 1st dimension but only index 1 of 2nd dimension
print(x[:, :, 1])

# Get all values of the 0 dimension but only the 1 index value of 1st and 2nd
# dimension
print(x[:, 1, 1])

# Get index 0 of oth dimension and all of 2nd dimenstion
print(x[:, 0, :])


