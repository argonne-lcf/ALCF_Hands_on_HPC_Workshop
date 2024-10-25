#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <vector>

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("4_buffer");

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

  //  _       _   _
  // |_)    _|_ _|_ _  ._
  // |_) |_| |   | (/_ |
  //

  // Crrate array
  std::vector<int> A(global_range);

  // Selectors determine which device kernels will be dispatched to.
  // Create your own or use `{cpu,gpu,accelerator}_selector`
  {
    // Create sycl buffer.
    // The buffer need to be destructed at the end of the scope to triger 
    // syncronization
    // Trivia: What happend if we create the buffer in the outer scope?
    sycl::buffer<int, 1> bufferA(A.data(), A.size());

    sycl::queue myQueue;
    std::cout << "Running on "
              << myQueue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    myQueue.submit([&](sycl::handler &cgh) {
      // Create an accesor for the sycl buffer. Trust me, use auto.
      auto accessorA = bufferA.get_access<sycl::access::mode::discard_write>(cgh);
      // Submit the kernel
      cgh.parallel_for<class hello_world>(
          sycl::range<1>(global_range), 
          [=](sycl::id<1> _) {
            //Need tp have a non void examplle
            accessorA[0];
#ifdef __SYCL_DEVICE_ONLY__
            auto idx = __spirv::initLocalInvocationId<1, sycl::id<1>>();
            accessorA[idx] = idx[0];
#endif
          }); // End of the kernel function
    });       // End of the queue commands
  }           // End of scope, wait for the queued work to stop.

  for (size_t i = 0; i < global_range; i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
