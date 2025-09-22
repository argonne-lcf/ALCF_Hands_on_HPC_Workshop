import dpnp as np

# Create array on default SYCL device (PVC GPU)
x = np.asarray([1, 2, 3])
print("Array x allocated on the device:", x.device)

# The queue associated with this array and future kernels acting on the array is carried with x
print("Array x is associated with SYCL queue:", x.sycl_queue)

# The pre-compiled kernel for np.sum(x) is submitted to queue of x
# The output array y is allocated on the same device as x and is associated with the queue of x
y = np.sum(x)
print("Result y is located on the device:", y.device)

# Expected output:
#	Array x allocated on the device: Device(level_zero:gpu:0)
#   Array x is associated with SYCL queue: <dpctl.SyclQueue at 0x151abb5fe440>
#	Result y is located on the device: Device(level_zero:gpu:0)
