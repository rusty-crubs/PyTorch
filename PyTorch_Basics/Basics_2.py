# Random Tensor

# Import PyTorch
import torch as torch

# Creating Random Tensors
RandomTensor = torch.rand(size=(3, 4))  # Row and column
print(RandomTensor, RandomTensor.dtype)

# Creating Random Image Tensor
# ([hight],[weight],[color_channels])
RandomImageTensor = torch.rand(size=(224, 224, 3))
print(RandomImageTensor.shape, RandomImageTensor.ndim)

# Zeros and Ones
# Sometimes you want to fill tensor with zeros and ones.
# This happens a lot with masking (like masking some of the values in one tensor to let a model know not
# to learn them).
# Creating a tensor full of zeros
print("Zeros:")
zeros = torch.zeros(size=(3, 4))
print(zeros, zeros.dtype)

# Creating a tensor full of open
print("Ones:")
ones = torch.ones(size=(3, 4))
print(ones, ones.dtype)

# Range and tensor
# Sometimes you might want range of numbers, such as 1 to 10 or 0 to 10
# can use torch.arange(start,end,step)
print("Range")
zero_to_ten = torch.arange(start=0, end=10, step=1)
print(zero_to_ten, zero_to_ten.dtype)

one_to_ten = torch.arange(start=1, end=11, step=1)
print(one_to_ten, one_to_ten.dtype)

# Creating ten zeros using range
ten_zeros = torch.zeros_like(input=zero_to_ten)
print(ten_zeros)

ten_ones = torch.ones_like(input=one_to_ten)
print(ten_ones)

# Tensor Datatype
print("\n Datatype in PyTorch Tensor")
# torch.float16 or torch.half
# torch.float32 Default datatype
# torch.float64 or torch.double

# default datatype for tensor in float32
print("-> Float32 ")
float_32_tensor = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=None,  # default to None, which is torch.float32, or what data type is passed
    device=None,  # defaults to None, which uses the default tensor type
    requires_grad=False  # if True operations performed on the tensor are
    # recorded
)
print(float_32_tensor, float_32_tensor.dtype, float_32_tensor.device)

print("\n_-> Float16 or half")
float_16_tensor = torch.tensor(
    [3.0, 6.0, 9.0],
    dtype=torch.half
)
print(float_16_tensor, float_16_tensor.dtype, float_16_tensor.device)

print("Example:\n")
# Create tensor
some_tensor = torch.rand(3, 4)

# find out deatails about it
print(some_tensor)
print("Shape of tensor ", some_tensor.shape)
print("Datatype of tensor", some_tensor.dtype)
print("Device of tensor ", some_tensor.device)
