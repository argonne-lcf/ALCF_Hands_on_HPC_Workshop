#include "argparse.hpp"
#include <sycl/sycl.hpp>

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |
  argparse::ArgumentParser program("2_parallel_for");

  program.add_argument("-g", "--global")
      .help("Global Range")
      .default_value(1)
      .action([](const std::string &value) { return std::stoi(value); });

  try {
    program.parse_args(argc, argv);
  } catch (const std::runtime_error &err) {
    std::cout << err.what() << std::endl;
    std::cout << program;
    exit(0);
  }

  const auto global_range = program.get<int>("-g");
  //  _                        _
  // |_) _. ._ _. | |  _  |   |_ _  ._
  // |  (_| | (_| | | (/_ |   | (_) |

  sycl::queue Q;
  std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

  // Similar to OMP PARALLEL FOR
  Q.parallel_for(
       // for(int idx=0; idx++; idx < global_range)
       global_range,
       [=](sycl::id<1> idx) {
         // Explicit cast because of printf shenaningan.
         sycl::ext::oneapi::experimental::printf("Hello, World! World rank %d\n",
                                                 static_cast<int>(idx));
       })
      .wait(); // Waiting on the event to finish.

  return 0;
}
