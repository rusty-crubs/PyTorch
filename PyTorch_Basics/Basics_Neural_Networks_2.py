# Finding the min, max, mean, sum etx (tensor aggregation)

# Import PyTorch and torch.neuralnetwork
import torch
# import torch.nn as nn

# Creating a tensor
# torch.arange(initial,final,steps)
x = torch.arange(0, 100, 10)
print(f"Input:{x}")
print(f"Input Dtype={x.dtype}")

# Finding min
print(f"Min of X:{torch.min(x)}")  # x.min() function could also be applied

# Finding Max
print(f"Max of X:{torch.max(x)}")  # x.max() function could also be applied

# Finding mean
print(f"Mean of X:{torch.mean(x.type(torch.float32))}")
# Note the torch.mean() function requires a tensor of float32 datatype to work

# Find the Sum
print(f"Sum of X:{torch.sum(x)}")  # x.sum() function could also be applied

# Find the position in tensor that has the minumum value with argmin()-. return
# index position of target tensor where the minimum value occur
print(f"Minimum value position in x:{x.argmin()}")
print(f"Value of X in zeroth position:{x[0]}")
# Find the position in tensor that has the minimum value with argmax() function
# return index position od target tensor where the maximum value occured
print(f"Maximum value position in x:{x.argmax()}")
print(f"Value of X in ninth position:{x[9]}")
