import torch
import numpy as np

# create simple scaler with .tensor method
scaler_valfloat = torch.tensor(65, dtype=torch.float64)
scaler_valint = torch.tensor(10, dtype=torch.int32)
print(scaler_valfloat)
print(scaler_valint)
