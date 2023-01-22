import torch
import numpy as np

# create simple scaler with .tensor method
scaler_valfloat = torch.tensor(65, dtype=torch.float64)
scaler_valint = torch.tensor(10, dtype=torch.int32)
print(scaler_valfloat)
print(scaler_valint)


# create vector of type int and float
vector_int = torch.tensor([4,5,6], dtype=torch.int32)
vector_float = torch.tensor([1,2,3], dtype=torch.float32)
print(vector_int)
print(vector_float)

# Create 1D tensors of 1s and 0s
ones_1d = torch.ones(5)
ones_1d = torch.ones(10)
print(ones_1d)

# Create 2D tensors of 1s and zeros
zeros_2d = torch.zeros((2,2))
print(zeros_2d)

# Create 3D tensor of 1s and zeros
ones_3d = torch.ones((3,2,2))
zeros_3d = torch.zeros((5,4,4))
print(ones_3d)
print(zeros_3d)


# Use arange and create 1D, 2D and 3D tensors
oneD_arange = torch.arange(10)
twoD_arange = torch.arange(4).view(2,2)
threeD_arange = torch.arange(8).view(2,2,2)
print(oneD_arange)
print(twoD_arange)
print(threeD_arange)

# Use linspace

oneD_linspace = torch.linspace(5,10,5)
twoD_linspace = torch.linspace(2,8,4).view(2,2)
print(twoD_linspace)

# Explore ones_like and zeros_like

twoD_arr = torch.zeros((5,5))
twoD_same = torch.zeros_like(twoD_arr)

twoD_arr_ones = torch.ones((3,3))
twoD_ones_same = torch.ones_like(twoD_arr_ones)
