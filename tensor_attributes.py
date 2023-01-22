import torch


# explore shape, size, type and other propertes of scaler, 1D, 2D and nD tensor

scaler_tensor = torch.tensor(65, dtype=torch.float32)

oneD_tensor = torch.arange(10,20)

twoD_tensor = torch.linspace(10,20,4).view(2,2)

threeD_tensor = torch.zeros((3,4,4))


# Shape of array
print(scaler_tensor.size())
print(scaler_tensor.shape)
print(oneD_tensor.size())
print(oneD_tensor.shape)
print(twoD_tensor.shape)
print(threeD_tensor.shape)

# Dimensions of array
print('Dimensions')
print(scaler_tensor.dim())
print(oneD_tensor.dim())
print(twoD_tensor.dim())
print(threeD_tensor.dim())

# dtype
print(scaler_tensor.dtype)
print(oneD_tensor.dtype)

# length of tensor
print(scaler_tensor.numel())
print(oneD_tensor.numel())
print(twoD_tensor.numel())
print(threeD_tensor.numel())

