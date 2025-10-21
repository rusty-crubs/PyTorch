# Reshaping, stacking, squeezing and unsqueexing tensor
# Reshaping - reshapes an input tensor to a defined shape
# View - Return a view of an input of certain shape but keep the same
# memory as the orginal tensor
# Stacking - combine multiple tensor on top of each other (vstack) or side by side (hstack)
# Squuze - remove all 1 dimensions from tensor
# Unsqeeze - add a 1 dimension to a target tensor
# Permut - Return a view of the input with dimension permuted (swapped) in a certain way

# Importing PyTorch
import torch
x = torch.arange(1.0, 10.0)
print(f"Value of X:{x},\nShape of X:{x.shape}")

# Add an extra dimension
print("Horizontal")
x_reshaped = x.reshape(1, 9)
print(f"Value of reshaped X:{x_reshaped}",
      f"\nShape of reshaped X:{x_reshaped.shape}")
print("\nVertical")
x_reshaped = x.reshape(9, 1)
print(f"Value of reshaped X:{x_reshaped}",
      f"\nShape of reshaped X:{x_reshaped.shape}")

# Change the view
z = x.view(1, 9)
print(f"View of X:{z},\nShape of X:{z.shape}")

# Changing z changes x (because a view of a tensor shares the same memory
# as the orginal)
z[:, 0] = 5
print(f"Value of Z:{z}")
print(f"Value of X:{x}")

# Stack tensors on top of each other
x_stacked = torch.stack(
    [x, x, x, x],
    dim=0
)
print(f"Stacked of x:{x_stacked}")

x = torch.rand(3, 3)
print(f"Value of X: {x}")
# Stack tensors on top of each other
x_stacked = torch.stack(
    [x, x, x, x],
    dim=2
)
print(f"Stacked of x:{x_stacked}")

x_vstacked = torch.vstack([x, x])
print(f"VStacked of X:{x_vstacked}")

x_hstacked = torch.hstack([x, x, x, x])
print(f"Hstacked of x:{x_hstacked}")

# Squeeze
print(f"Previous input:{x_reshaped} and its shape:{x_reshaped.shape}")
x_squeeze = x_reshaped.squeeze()
print("New tensor\n")
print(f"Its squeeze: {x_squeeze}, and its shape: {x_squeeze.shape}")

# Won't work in row x column
x = torch.rand(3, 3)
print(f"Value of inputs:\n{x}")
print(f"Shape of X:{x.shape}")

x_squeezed = x.squeeze()
print(f"New tensor:\n{x_squeezed}")
print(f"Shape:\n{x_squeezed.shape}")

# torch.unsqueeze
print(f"Previous target:{x_squeeze}")
print(f"Shape of taget: {x_squeeze.shape}")
x_unsqueeze = x_squeeze.unsqueeze(dim=0)
print(f"New Tensor: {x_unsqueeze}")
print(f"Shape:{x_unsqueeze.shape}")

# torch.permute - rearrages the dimension of a target tensor
# in a specified order
input = torch.randn(size=(244, 244, 5))
print(f"Shape:{input.shape}")
# Permute the orginal tensor to rearrange the axis (or dim) order
input_permuted = input.permute(2, 0, 1)
print(f"New Shape:{input_permuted.shape}")
