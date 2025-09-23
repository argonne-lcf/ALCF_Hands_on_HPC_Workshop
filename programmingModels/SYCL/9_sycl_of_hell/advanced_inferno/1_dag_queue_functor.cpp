#include "argparse.hpp"
#include <sycl/sycl.hpp>
#include <iostream>
#include <numeric>
#include <vector>

template <typename TAccessorW, typename TAccessorR>
class memcopy_kernel {
public:
  memcopy_kernel(TAccessorW accessorW_, TAccessorR accessorR_)
      : accessorW{accessorW_}, accessorR{accessorR_} {}
  void operator()(sycl::item<1> idx) const {
    accessorW[idx.get_id()] = accessorR[idx.get_id()] + 1;
  }

private:
  TAccessorW accessorW;
  TAccessorR accessorR;
};

void f_copy(sycl::handler &cgh, int global_range,
            sycl::buffer<int, 1> bufferW,
            sycl::buffer<int, 1> bufferR) {
  auto accessorW = bufferW.get_access<sycl::access::mode::discard_write>(cgh);
  auto accessorR = bufferR.get_access<sycl::access::mode::read>(cgh);
  cgh.parallel_for(sycl::range<1>(global_range),
                   memcopy_kernel(accessorW, accessorR));
}

int main(int argc, char **argv) {

  //  _                ___
  // |_) _. ._ _  _     |  ._  ._     _|_
  // |  (_| | _> (/_   _|_ | | |_) |_| |_
  //
  argparse::ArgumentParser program("1_dag_queue_functor");

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

  std::cout << "Running a linear dag. Output should start at 3" << std::endl;
  std::vector<int> A(global_range);
  std::iota(std::begin(A), std::end(A), 0);

  //   _      __
  //  | \ /\ /__
  //  |_//--\\_|

  // Selectors determine which device kernels will be dispatched to.
  sycl::queue Q;
  std::cout << "Running on "
            << Q.get_device().get_info<sycl::info::device::name>()
           << "\n";
    
  // A -> B -> C -> A
  // Create buffers
  sycl::buffer  bufferA(A);
  // Empty buffer
  sycl::buffer<int, 1> bufferB(global_range);
  sycl::buffer<int, 1> bufferC(global_range);

  Q.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                     bufferB, bufferA));
  Q.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                     bufferC, bufferB));
  Q.submit(std::bind(f_copy, std::placeholders::_1, global_range,
                     bufferA, bufferC));
 
  sycl::host_accessor AccA { bufferA, sycl::read_only };
  for (size_t i = 0; i < global_range; i++) {
    assert ( AccA[i] == i + 3);
    std::cout << "A[ " << i << " ] = " << AccA[i] << std::endl;
  }
  return 0;
}
