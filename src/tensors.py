import torch
import numpy as np

# Tensor Initialization
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# From NumPy Array
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# From Another Tensor
x_ones = torch.ones_like(x_data)  # retains the properties of x_data
print(f"Ones Tensor: \n{x_ones}\n")
x_rand = torch.rand_like(x_data, dtype=torch.float)  # overrides the datatype of x_data
print(f"Random Tensor: \n{x_rand}\n")

# With Random or Constant Values
shape = (2, 3)
rand_tensor = torch.rand(shape)
print(f"Random Tensor of shape {shape}: \n{rand_tensor}\n")
ones_tensor = torch.ones(shape)
print(f"Ones Tensor of shape {shape}: \n{ones_tensor}\n")
zeros_tensor = torch.zeros(shape)
print(f"Zeros Tensor of shape {shape}: \n{zeros_tensor}\n")

# Tensor Attributes
tensor = torch.rand(3, 4)
print(f"Tensor: \n{tensor}\n")
print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}\n")

# Tensor Operations

# Move tensor to GPU if available
if torch.cuda.is_available():
    tensor = tensor.to('cuda')
    print(f"Tensor moved to GPU: {tensor.device}\n")
    
# Indexing and Slicing
tensor = torch.arange(1, 13).reshape(3, 4)
print(f"Original Tensor: \n{tensor}\n")
print(f"First row: {tensor[0, :]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[:, -1]}\n")

# Concatenation
t1 = torch.rand(2, 3)
t2 = torch.rand(2, 3)
concat_dim0 = torch.cat([t1, t2], dim=0)
print(f"Concatenated along dim 0: \n{concat_dim0}\n")
concat_dim1 = torch.cat([t1, t2], dim=1)
print(f"Concatenated along dim 1: \n{concat_dim1}\n")

# Multiplication

# Element-wise multiplication
mul_elem = t1 * t2
print(f"Element-wise multiplication: \n{mul_elem}\n")

# Matrix multiplication
t3 = torch.rand(3, 2)
matmul = torch.matmul(t1, t3)
print(f"Matrix multiplication: \n{matmul}\n")

# In-place operations | Bad Practice.
tensor = torch.rand(2, 2)
print(f"Original Tensor before in-place operation: \n{tensor}\n")
tensor.add_(5)
print(f"Tensor after in-place addition: \n{tensor}\n")


# Tensor to NumPy Array
tensor = torch.ones(5)
np_array = tensor.numpy()
print(f"Tensor: {tensor}")
print(f"Converted NumPy Array: {np_array}\n")

# Modifying the tensor will also modify the NumPy array
tensor.add_(1)
print(f"Modified Tensor: {tensor}")
print(f"NumPy Array after Tensor modification: {np_array}\n")

# NumPy Array to Tensor
np_array = np.ones(5)
tensor = torch.from_numpy(np_array)
print(f"NumPy Array: {np_array}")
print(f"Converted Tensor: {tensor}\n")

# Modifying the NumPy array will also modify the tensor
np_array += 1
print(f"Modified NumPy Array: {np_array}")
print(f"Tensor after NumPy Array modification: {tensor}\n")# Tensor and NumPy Interoperability
    
