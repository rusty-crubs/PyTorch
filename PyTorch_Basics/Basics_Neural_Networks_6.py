# import PyTorch
import torch

# Create a random tensor with shape(7,7)
seed = 0
torch.manual_seed(seed)
tensor = torch.rand(7,7)
print(f"Tensor:\n{tensor}")

# Matrix Multiplication with another random tensor with shape (1,7)
torch.manual_seed(seed)
random_tensor = torch.rand(1,7)
print(f"Tensor:\n{random_tensor}")

NewTotalTensor = torch.matmul(random_tensor,tensor)
print(f"Multiplication of 2 random matrix:\n{NewTotalTensor}")

maximum = NewTotalTensor.max()
maximum_position = NewTotalTensor.argmax()
minimum = NewTotalTensor.min()
minimum_position = NewTotalTensor.argmin()
print(f"Maximum Value: {maximum}")
print(f"Position: {maximum_position}")
print(f"Minimum Value: {minimum}")
print(f"Position: {minimum_position}")


