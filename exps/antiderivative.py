#%%
import wget
import os 
import deeponets
import numpy as np

#%%
wget.download("https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_train.npz")
wget.download("https://github.com/mroberto166/CAMLab-DLSCTutorials/raw/main/antiderivative_aligned_test.npz")


# %%

dataset_train = np.load("antiderivative_aligned_train.npz", allow_pickle=True)

branch_train = dataset_train["X"][0]
trunk_train = dataset_train["X"][1]
outputs_train = dataset_train["y"]

print('branch', branch_train.shape)
print('trunk', trunk_train.shape)
print('output', outputs_train.shape)  
# %%

trunk_train.unsqueeze(0).repeat(150).shape

# %%
