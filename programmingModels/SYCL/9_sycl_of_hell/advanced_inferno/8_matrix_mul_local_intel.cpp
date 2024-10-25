/*
Copyright (C) 2019 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

SPDX-License-Identifier: MIT
*/
#include <sycl/sycl.hpp>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>

template <typename T, int dimensions>
using local_accessor =
    sycl::accessor<T, dimensions, sycl::access::mode::read_write, sycl::access::target::local>;

int main(int argc, char *argv[]) {
  // Set up SYCL device and queue.
  sycl::device dev(sycl::gpu_selector_v);
  sycl::queue q(dev);

  // Initialize input and output memory on the host
  const uint32_t N = 1024;
  const uint32_t B = 4;
  std::vector<float> a(N * N), b(N * N), c(N * N);
  std::default_random_engine rng(42);
  std::uniform_real_distribution<float> dist(0.0, 1.0);
  std::generate(a.begin(), a.end(), [&]() { return dist(rng); });
  std::generate(b.begin(), b.end(), [&]() { return dist(rng); });
  std::fill(c.begin(), c.end(), 0);

  {
    // Create SYCL buffers associated with input/output
      sycl::buffer<float, 2> a_buf(a.data(), sycl::range<2>(N, N)),
        b_buf(b.data(), sycl::range<2>(N, N)), c_buf(c.data(), sycl::range<2>(N, N));

      q.submit([&](sycl::handler &cgh) {
      auto a = a_buf.get_access<sycl::access::mode::read>(cgh);
      auto b = b_buf.get_access<sycl::access::mode::read>(cgh);
      auto c = c_buf.get_access<sycl::access::mode::read_write>(cgh);

      auto a_tile = local_accessor<float, 2>(sycl::range<2>(B, B), cgh);
      auto b_tile = local_accessor<float, 2>(sycl::range<2>(B, B), cgh);

      sycl::range<2> global(N, N);
      sycl::range<2> local(B, B);
      cgh.parallel_for<class matrix_mul>(
          sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int j = item.get_global_id(0);
            int i = item.get_global_id(1);

            int lj = item.get_local_id(0);
            int li = item.get_local_id(1);

            for (int kb = 0; kb < N / B; ++kb) {
              // Load tiles of A and B matrices into local memory
              a_tile[lj][li] = a[j][kb * B + li];
              b_tile[lj][li] = b[kb * B + lj][i];

              // Wait for load into local memory to complete
              // Barrier synchronizes all work-items in this work-group
              item.barrier(sycl::access::fence_space::local_space);

              // Compute matrix multiply using results in local memory
              for (int k = 0; k < B; ++k) {
                c[j][i] += a_tile[lj][k] * b_tile[k][li];
              }

              // Ensure all work-items are done before overwriting local memory
              // Barrier synchronizes all work-items in this work-group
              item.barrier(sycl::access::fence_space::local_space);
            }
          });
    });
  }

  // Check that all outputs match serial execution.
  bool passed = true;
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < N; ++i) {
      float gold = 0;
      for (int k = 0; k < N; ++k) {
        gold += a[j * N + k] * b[k * N + i];
      }
      if (std::abs(gold - c[j * N + i]) / gold > 1.0E-06) {
        passed = false;
      }
    }
  }
  std::cout << ((passed) ? "SUCCESS" : "FAILURE") << std::endl;
  return 0;
}
