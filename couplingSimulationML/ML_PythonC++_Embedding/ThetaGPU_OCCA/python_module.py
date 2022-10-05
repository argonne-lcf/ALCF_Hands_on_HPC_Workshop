print("Within Python Module File")

import os,sys
HERE = os.getcwd()
sys.path.insert(0,HERE)

import numpy as np
import cupy
from cupy.cuda import memory
import matplotlib.pyplot as plt

data_array = cupy.zeros(shape=(2001,258)) # matches the number of timesteps in the main solver
x = np.arange(start=0,stop=2.0*3.1415926,step=2.0*3.1415926/256)
iternum = 0

def collection_func(input_array):
    global data_array,iternum
    b = cupy.ndarray(
                input_array.__array_interface__['shape'][0],
                cupy.dtype(input_array.dtype.name),
                cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                           input_array.__array_interface__['data'][0],
                                           input_array.size,
                                           input_array,
                                           0), 0),
                strides=input_array.__array_interface__['strides'])
    data_array[iternum,:] = b[:]
    iternum+=1
    return None

def analyses_plotField():
    global data_array, x

    plt.figure()
    for i in range(0,cupy.shape(data_array)[0],400):
        y = cupy.asnumpy(data_array[i,1:-1]) 
        plt.plot(x,y,label='Timestep '+str(i))    
    plt.legend()
    plt.xlabel('x')
    plt.xlabel('u')
    plt.title('Field evolution')
    plt.savefig('Field_evolution.png')
    plt.close()

def analyses_SVD():
    global data_array, x

    # Perform an SVD on device
    data_array = data_array[:,1:-1]
    print('Performing SVD')
    u,s,v = cupy.linalg.svd(data_array,full_matrices=False)

    # Plot SVD eigenvectors
    vh = cupy.asnumpy(v)
    plt.figure()
    plt.plot(x, vh[0,:],label='Mode 0')
    plt.plot(x, vh[1,:],label='Mode 1')
    plt.plot(x, vh[2,:],label='Mode 2')
    plt.legend()
    plt.title('SVD Eigenvectors')
    plt.xlabel('x')
    plt.xlabel('u')
    plt.savefig('SVD_Eigenvectors.png')
    plt.close()

#DONE 
