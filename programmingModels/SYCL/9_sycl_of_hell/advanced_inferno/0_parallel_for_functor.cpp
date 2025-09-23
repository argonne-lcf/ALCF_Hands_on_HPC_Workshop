#include "argparse.hpp"
#include <sycl/sycl.hpp>

class generator_kernel_hw {

public:
  generator_kernel_hw(sycl::stream cout) : m_cout(cout) {}

  void operator()(sycl::id<1> idx) const {
    m_cout << "Hello, World Functor: World rank " << idx << sycl::endl;
  }

private:
  sycl::stream m_cout;
};

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //                           |

  argparse::ArgumentParser program("0_parallel_for_functor");

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

  Q.submit([&](sycl::handler &cgh) {
      sycl::stream cout(1024, 256, cgh);
      generator_kernel_hw kernel{cout};
      cgh.parallel_for(sycl::range<1>(global_range), kernel);
  }).wait(); // End of the queue commands
  return 0;
}
