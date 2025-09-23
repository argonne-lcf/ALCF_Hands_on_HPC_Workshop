#include <sycl/sycl.hpp>

int main() {
  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  sycl::program p(Q.get_context());
  // build_with_source may take an aditioanl argument to pass compile flags
  p.build_with_source(R"EOL(__kernel void hello_world() {printf("Hello world\n");} )EOL");

  //  _               _
  // / \ ._   _  ._  /  |    |/  _  ._ ._   _  |
  // \_/ |_) (/_ | | \_ |_   |\ (/_ |  | | (/_ |
  //     |
 Q.submit([&](sycl::handler &cgh) {
    // Will launch an opencl kernel
    // Use cgh.set_args($acceors) if required by your kermel
    cgh.single_task(p.get_kernel("hello_world"));
 }).wait();
 return 0;
}
