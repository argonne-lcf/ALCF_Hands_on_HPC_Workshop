#include <sycl/sycl.hpp>

int main() {
  //         __                     ___
  //  /\    (_  o ._ _  ._  |  _     |  _.  _ |
  // /--\   __) | | | | |_) | (/_    | (_| _> |<
  //                    |

  sycl::queue Q;
  std::cout << "Running on " << Q.get_device().get_info<sycl::info::device::name>() << "\n";

  // Submit a one work item (a single task) to the GPU using a lambda
  // Queue submission are asyncrhonous (similar to OpenMP nowait)
  Q.single_task([=]() { sycl::ext::oneapi::experimental::printf("Hello, World!\n"); });
  // wait for all queue submissions to complete
  Q.wait();

  return 0;
}
