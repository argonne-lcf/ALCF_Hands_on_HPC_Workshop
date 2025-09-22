import dpctl

# List the SYCL platform including information on the devices on the node
print("List the SYCL platform")
dpctl.lsplatform(verbosity=1)

# Get the number of GPU devices on the node
num_devices = dpctl.get_num_devices(device_type="gpu")

# List the GPU devices on the node
device_list =  dpctl.get_devices(device_type="gpu")

# Print the number of GPU devices and the properties of each GPU
print(f"\nFound {num_devices} GPU devices with properties:")
for device in device_list:
    print(f"\t{device}")

# Print the number of CPU devices
print("\nFound CPU devices: ", dpctl.has_cpu_devices())

# Expected output:
#   List the SYCL platform
#   Platform  0 ::
#       Name        Intel(R) oneAPI Unified Runtime over Level-Zero
#       Version     1.6
#       Vendor      Intel(R) Corporation
#       Backend     ext_oneapi_level_zero
#       Num Devices 12
#
#   Found 12 GPU devices
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153eccfd5830>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153eccfd58b0>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153eccfd5870>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153ec9f2d870>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153ec9fa93f0>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153ecd047d70>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153e409619b0>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153e409b0330>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153e409b03f0>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153ec9fa9d30>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153e409b0370>
#       <dpctl.SyclDevice [backend_type.level_zero, device_type.gpu,  Intel(R) Data Center GPU Max 1550] at 0x153e409b0c30>
#
#   Found CPU devices:  False
