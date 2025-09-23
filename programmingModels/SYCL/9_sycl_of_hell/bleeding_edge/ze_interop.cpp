#include <sycl/sycl.hpp>
#include <level_zero/ze_api.h>
#include <CL/sycl/backend/level_zero.hpp>

namespace sycl = cl::sycl;

int main() {

  sycl::queue Q;
  sycl::device dev = Q.get_device();

  std::cout << "Device Name "
            << dev.get_info<sycl::info::device::name>()
            << std::endl;

  auto hDevice = dev.get_native<sycl::backend::level_zero>();

  ze_device_properties_t deviceProperties = {};
  zeDeviceGetProperties(hDevice, &deviceProperties);

  std::cout << "Device number slices "
            << deviceProperties.numSlices
            << std::endl;
}
