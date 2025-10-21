# PyTorch Tensors
# Importing PyTorch
import torch as torch

# Scalar Tensor
print("Scalar Tensors")
scalar = torch.tensor(4)
print(scalar)
print("dimension", scalar.ndim)
scalar.item()
print("Shape: ", scalar.shape)

# Vector Tensor
print("Vector Tensors")
vector = torch.tensor([7, 7])
print(vector)
print("dimensions: ", vector.ndim)
print("Shape: ", vector.shape)

# Matrix Tensor
print("Matrix Tensors")
Matrix = torch.tensor([
    [7, 7],
    [8, 9]
])
print(Matrix)
print("dimension: ", Matrix.ndim)
print(Matrix.shape)

# Tensor Tensor
print("Tensor")
TENSOR = torch.tensor([[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]])
print(TENSOR)
print("dimension", TENSOR.ndim)
print("Shape", TENSOR.shape)
