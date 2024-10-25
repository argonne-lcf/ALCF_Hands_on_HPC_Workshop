#include <sycl/sycl.hpp>
#include <iostream>
#include <stdio.h>

#define WORKSIZE 256
#define WORKITEM 64

int main(int argc, char **argv) {
  sycl::queue q(sycl::gpu_selector_v);

  q.submit([&](sycl::handler &cgh) {
    auto acc = sycl::accessor<int, 1, sycl::access::mode::read_write,
                              sycl::access::target::local>(
        sycl::range<1>(WORKSIZE), cgh);

    cgh.parallel_for<class kernel1>(
        sycl::nd_range<1>(sycl::range<1>(WORKSIZE), sycl::range<1>(WORKITEM)),
        [=](sycl::nd_item<1> i) {
          int x = i.get_global_linear_id();
          int y = i.get_local_linear_id();
          acc[y] = x;
          i.barrier();
        });
  });
  q.wait();

  return 0;
}
