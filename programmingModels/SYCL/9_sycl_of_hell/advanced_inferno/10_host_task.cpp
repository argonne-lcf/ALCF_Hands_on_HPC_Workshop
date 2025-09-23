#include <sycl/sycl.hpp>
// Inspired by Victor (vanisimov@anl.gov) example.

int main() {
  sycl::queue  Q;
  std::cout << "Device Name: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  int *a = sycl::malloc_shared<int>(1, Q);
  a[0] = 0;
  
  // We don't wait for this kernel, to show that one can use host_task as callback on kernel finish
  auto e = Q.single_task( [=]() { a[0] = 1; } );

  Q.submit([&](sycl::handler &cgh) {
      // This function will be scheduled after the kernel who created the event is finished
      // Note that if one create a 'in_order queue', one can omit this `depend_on` as it will be implicit.
      cgh.depends_on(e);
      cgh.host_task( [=]() {
         //Note that we are using `std::cout` and `std::cout` is an host function!
         // We are running on the  host. So one can do anything they want
         // (take a lock, call a native function - cublas - and so one)
         std::cout << "Host Task a[0] " <<  a[0] << std::endl;
         std::cout << "Expected 1" << std::endl;
         // We are sure that this will be correct due to the dependencies
         assert(a[0] == 1);
      });
  }).wait();
}
