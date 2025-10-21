# Tensor:
# Similar to array or matrix
# Building block of neural network
# tensor.shape to display the shape of matrix or list
# tensor.dtype to show the data type of the tensor which help for debugging
# @ for matrix multiplication Row X Colume

# Tenser are PyTorch's core data structure and the foundation of deep learning.
# They're similar to NumPy arrays but have unique features.

# Here we have Python list named tempuratures containing daily readings from
# two weather
# stations. Lets try converting this into tensor!!

# import PyTorch
import torch as torch
temperatures = [[72, 75, 78], [70, 73, 76]]

# Create Tensor for temperatures
temp_tensor = torch.tensor(temperatures)

print(temp_tensor)
