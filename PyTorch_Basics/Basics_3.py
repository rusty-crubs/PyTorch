# Importing PyTorch
import torch

# Manipulating Tensors (tensor operations)
tensor = torch.tensor([1, 2, 3])

# Addition
print("Addition")
print(tensor + 10)

# subtraction
print("subtraction")
print(tensor - 10)

# multiplication
print("multiplication")
tensor = torch.multiply(tensor, 10)
print(tensor)

# division
print("Division")
tensor = torch.divide(tensor, 10)
print(tensor)

# matrix multiplication
# There are two main rules that performing matrix
# multiplication needs to satisfy:
# 1. The inner dimentions must match meaning
# (3,2) @ (3,2) won't work
# (2,3) @ (3,2) will work
# (3,2) @ (2,3) will work
# 2. The resulting matrix has the shape of the outer dimension
# (2,3) @ (3,2) --> (2,2)
# vector x vector
print("Matrix Multiplication")
print("vector x vector")
tensor1 = torch.randn(3)
tensor2 = torch.randn(3)
print(torch.matmul(tensor1, tensor2), torch.matmul(tensor1, tensor2).size())
print(tensor1 @ tensor2)

# matrix x vector
print("Matrix x vector")
tensor1 = torch.randn(3, 4)
tensor2 = torch.randn(4)
print(torch.matmul(tensor1, tensor2).size())

# batch matrix x boardcasted vector
print("batch matrix x boardcasted vector")
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4)
print(torch.matmul(tensor1, tensor2).size())

# batched matrix x batched matrix
print("batched matrix x batched matrix")
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(10, 4, 5)
print(torch.matmul(tensor1, tensor2).size())

# batched matrix x boardcasted matrix
print("batched matrix x boardcasted matrix")
tensor1 = torch.randn(10, 3, 4)
tensor2 = torch.randn(4, 5)

# torch.mm() Shortcut version of torch.matmul()
print(torch.matmul(tensor1, tensor2).size())

tensor1 = torch.tensor(
    [
        [1, 2],
        [3, 4],
        [5, 6]
    ],
    dtype=None
)
print("Shape", tensor1.shape)

tensor2 = torch.tensor(
    [
        [7, 10],
        [8, 11],
        [9, 12]
    ],
    dtype=None
)
print("Shape", tensor2.shape)
print(tensor2.T)
print(torch.mm(tensor1, tensor2.T))  # tensor.T is the transpose of tensor
