#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <vector>

// Note: please don't use std::atomic_ref<T> in the device-code
template <typename T>
using relaxed_atomic_ref =
    sycl::atomic_ref< T,
		      sycl::memory_order::relaxed,
		      sycl::memory_scope::device,
		      sycl::access::address_space::global_space>;

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("4_buffer_atomic.cpp");

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
  int A = 0;
  int A_atom = 0;
  {
    // Create sycl buffers.
    sycl::buffer<int> bufferA(&A, sizeof(int));
    sycl::buffer<int> bufferA_atom(&A_atom, sizeof(int));

    sycl::queue Q;
    std::cout << "Running on "
              << Q.get_device().get_info<sycl::info::device::name>()
              << "\n";

    // Create a command_group to issue command to the group
    Q.submit([&](sycl::handler &cgh) {
      sycl::accessor accessorA{bufferA, cgh, sycl::read_write};
      sycl::accessor accessorA_atom(bufferA_atom, cgh, sycl::read_write);
      // Legacy version:
      //auto accessorA_atom = bufferA_atom.get_access<sycl::access::mode::atomic>(cgh);
      cgh.parallel_for(
          sycl::range<1>(global_range), [=](auto _) {
            accessorA[0] +=1 ;
            auto atm = relaxed_atomic_ref<int>(accessorA_atom[0]);
            atm.fetch_add( 1 );
          }); 
    });   
  }      

  std::cout << "Counter incrememented " << global_range << " time " << std::endl;
  std::cout << "Atomic Increment:" << A_atom << std::endl;
  assert( A_atom == global_range);
  std::cout << "Non Atomic Increment " << A << std::endl;
  return 0;
}
