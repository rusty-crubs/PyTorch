# PyTorch tensor and NumPy
import torch
import numpy as np

array = np.arange(1.0, 8.0)
tensor = torch.from_numpy(array)
print(f"NumPy array:{array}")
print(f"Numpy Datatype:{array.dtype}")
print(f"Tensor:{tensor}")
print(f"Tensor dtype: {tensor.dtype}")


# Random seed
RANDOM_SEED = 4
torch.manual_seed(RANDOM_SEED)
random_tensor_A = torch.rand(3, 4)
torch.manual_seed(RANDOM_SEED)
random_tensor_B = torch.rand(3, 4)

print(f"First Random target:\n{random_tensor_A}")
print(f"Second Random target:\n {random_tensor_B}")
print(f"Comparision:\n {random_tensor_A == random_tensor_B}")
