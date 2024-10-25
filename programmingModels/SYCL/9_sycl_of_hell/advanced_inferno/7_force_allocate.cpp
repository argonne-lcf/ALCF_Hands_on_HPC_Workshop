#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

// How to transform this function into a variadic one
template <sycl::access::target Q, typename T, int I>
void force_allocate(sycl::buffer<T, I> buffer, sycl::queue myQueue) {

  // Create a device accesors and use it. This will force the allocation
  myQueue.submit([&](sycl::handler &cgh) {
    sycl::accessor<T, I, sycl::access::mode::write, Q> accessorA(
        buffer, cgh, buffer.byte_size());
    cgh.single_task<class allocate>([=]() { accessorA[0]; });
  });
  myQueue.wait();
}

int main(int argc, char **argv) {
  argparse::ArgumentParser program("7_force_allocate.cpp");
  
  program.add_argument("-g","--global")
   .help("Global Range")
   .default_value(1)
   .action([](const std::string& value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  const auto global_range = program.get<int>("-g");

  std::vector<int> A(global_range);

  {
    sycl::queue myQueue(sycl::gpu_selector_v);

    // Create buffer.
    sycl::buffer<int, 1> bufferA(A.data(), global_range);

    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Force the allocation of the buffer.

    force_allocate<sycl::access::target::device>(bufferA, myQueue);

  } // End of scope, wait for the queued work to stop.
  return 0;
}
