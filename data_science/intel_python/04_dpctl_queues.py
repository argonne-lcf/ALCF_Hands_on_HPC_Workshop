import dpnp as dp
import dpctl

# Get the list of devices
devices = dpctl.get_devices()
num_devices = len(devices)
print(f'Found {num_devices} GPU devices') 

# Create a SYCL queue for each of the device
_queues = [None] * num_devices
for device_id in range(num_devices):
    _queues[device_id] = dpctl.SyclQueue(devices[device_id])

# Allocate an array on each of the devices with the queues
def func(device_id=0):
    arr = dp.ndarray([0,1,2],sycl_queue=_queues[device_id])
    return arr

for device_id in range(num_devices):
    print(func(device_id).device)

# Expected output:
#	Found 12 GPU devices
#	Device(level_zero:gpu:0)
#	Device(level_zero:gpu:1)
#	Device(level_zero:gpu:2)
#	Device(level_zero:gpu:3)
#	Device(level_zero:gpu:4)
#	Device(level_zero:gpu:5)
#	Device(level_zero:gpu:6)
#	Device(level_zero:gpu:7)
#	Device(level_zero:gpu:8)
#	Device(level_zero:gpu:9)
#	Device(level_zero:gpu:10)
#	Device(level_zero:gpu:11)

