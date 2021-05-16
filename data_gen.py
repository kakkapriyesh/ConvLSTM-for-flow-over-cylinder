from numpy import *
import scipy.linalg
import numpy as np
import matplotlib.pyplot as plt
from gridGen import grid,gridPlot,spacing,uGrid
import torch as th
import torch
import h5py as h

TD1=torch.from_numpy(np.load('train_data_250_8_RE500.npy'))
TD2=torch.from_numpy(np.load('train_data_250_8_RE1000.npy'))
TD3=torch.from_numpy(np.load('train_data_250_8_RE1500.npy'))
TD4=torch.from_numpy(np.load('train_data_250_8_RE2500.npy'))
TD5=torch.from_numpy(np.load('train_data_250_8_RE5000.npy'))
TD6=torch.from_numpy(np.load('train_data_250_8_RE7500.npy'))
TD7=torch.from_numpy(np.load('train_data_250_8_RE8500.npy'))
TD8=torch.from_numpy(np.load('train_data_250_8_RE10000.npy'))



TD = torch.cat([TD1, TD2,TD3,TD4,TD5,TD6,TD7,TD8], dim=0)
n=int(TD.shape[0]/10)
print("The shape of input data is",TD.shape)
psiMat1=np.zeros((n,10,64,64))
k=0
for i in range(0,TD.shape[0],10):
    psiMat1[k,:,:,:]=TD[i:i+10,:,:]
    k=k+1
print("Shape of data is",psiMat1.shape)
input=np.zeros((n,5,64,64))
target=np.zeros((n,5,64,64))
k1=0
for i in range(n):
    input[i,:,:,:]=psiMat1[i,0:5,:,:]
    target[i,:,:,:] = psiMat1[i,5:10,:,:]
print("Shape of input data is",input.shape)
print("Shape target data is",target.shape)



#input=(input - np.min(input)) / (np.max(input) - np.min(input))
print("maxi",np.max(input))
print("mini",np.min(input))
#target=(target - np.min(target)) / (np.max(target) - np.min(target))
print("maxt",np.max(input))
print("mint",np.min(input))

inputs = th.from_numpy(input).contiguous()
targets = th.from_numpy(target).contiguous()
inputs=th.unsqueeze(inputs,2)
targets=th.unsqueeze(targets,2)

print("Shape of input data is",inputs.shape)
print("Shape of target data is",targets.shape)


file = open("dataset_input_8_cases_lid.npy", "wb")
np.save('dataset_input_8_cases_lid.npy', inputs)
file.close

file = open("dataset_target_8_cases_lid.npy", "wb")
np.save('dataset_target__8_cases_lid.npy', targets)
file.close

#plotting stuff
rows=2
cols=3
fig = plt.figure()
axes=[]

for idx in range(10):
    b=input[idx*25+23,1,:,:]
    axes.append( fig.add_subplot(rows, cols, idx+1) )


    plt.contour(flipud(b), 10, extend='both')
fig.tight_layout()
plt.show()
axest=[]
for idx in range(3):
    c=target[5,idx,:,:]
    axes.append( fig.add_subplot(rows, cols, idx+1) )


    plt.contour(flipud(c), 10, extend='both')
fig.tight_layout()
plt.show()
for i in range(2):
    psiMat=input[0,i,:,:]
    print(i)
    cs = plt.contour(flipud(psiMat), 10, extend='both')
    cs.cmap.set_over('red')
    cs.cmap.set_under('blue')
    cs.changed()
    plt.pause(0.002)
    plt.clf()

