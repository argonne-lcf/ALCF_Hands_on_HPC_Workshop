import dpnp as dp
import dpctl

devices = dpctl.get_devices()
num_devices = len(devices)
print(f'Found {num_devices} GPU devices') 

_queues = [None] * num_devices
for device_id in range(num_devices):
    _queues[device_id] = dpctl.SyclQueue(devices[device_id])

def func(device_id=0):
    arr = dp.ndarray([0,1,2],sycl_queue=_queues[device_id])
    return arr

for device_id in range(num_devices):
    print(func(device_id).device)

