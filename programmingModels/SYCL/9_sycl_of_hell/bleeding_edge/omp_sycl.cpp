#include <sycl/sycl.hpp>
#include <stdio.h>
 
int main() {
  int resSycl = 1234;
  {
    sycl::buffer<int, 1> resBuffer(&resSycl, sycl::range<1>(1));
    sycl::queue().submit([&](sycl::handler &cgh) {
        auto resAcc = resBuffer.get_access<sycl::access::mode::write>(cgh);
        cgh.single_task<class X>([=]() {  resAcc[0] = 1; });
      });
  }
 
  int resOmp = 4321;
#pragma omp target map(from: resOmp)
  resOmp = 2;
 
  printf("resSycl = %d\n", resSycl);
  printf("resOmp = %d\n", resOmp);
  return 0;
}
