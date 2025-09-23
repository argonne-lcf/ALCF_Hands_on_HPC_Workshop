#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <vector>

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("7_allocator_usm");

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

  sycl::queue Q;

  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
            << "\n";

  // Create custom shared usm allocator and use it
  sycl::usm_allocator<float, sycl::usm::alloc::shared> allocator(Q);
  std::vector<float, decltype(allocator)> A(global_range, allocator);

  // A vector is not trivialy copyable
  auto *A_p = A.data();
  // One case use sycl::malloc_device is they prefere to work with row pointer
  // See example in advended infermo
  
  //Short syntax
  // USM is shorted but you need to handle the data-dependency ordered yourself
  // "Naive" way is to put `wait` after each enqueue.
  Q.parallel_for(global_range, [=](sycl::id<1> idx) { A_p[idx] = idx; }).wait();

  //We can use your vector as usual
  for (size_t i = 0; i < A.size(); i++)
    std::cout << "A[ " << i << " ] = " << A[i] << std::endl;
  return 0;
}
