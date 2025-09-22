import dpnp as np 

x = np.asarray([1, 2, 3])
print("Array x allocated on the device:", x.device)

y = np.sum(x)
print("Result y is located on the device:", y.device)

