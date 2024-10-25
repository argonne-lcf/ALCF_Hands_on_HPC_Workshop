/***************************************************************************
 *
 *  Copyright (C) 2016 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's ComputeCpp SDK
 *
 *  simple-private-memory.cpp
 *
 *  Description:
 *    Sample showing the use of the hierarchical API in SYCL.
 *
 **************************************************************************/

#include <sycl/sycl.hpp>

class PrivateMemory;

/* Helper function to compute a globalID from a group and item in a
 * hierarchical parallel_for_work_item context */
static inline sycl::id<1> get_global_id(sycl::group<1>& group,
					sycl::item<1>& item) {
  auto groupID = group.get_group_id();
  auto localR = item.get_range();
  auto localID = item.get_id();
  return sycl::id<1>(groupID[0] * localR[0] + localID[0]);
}

/* This sample showcases the syntax of the private_memory interface. */
int main() {
  int ret = 0;
  constexpr size_t nItems = 64;
  constexpr size_t nLocals = 16;
  int data[nItems] = {0};

  /* Any data on the device will be copied back to the host
   * after the block ends. */
  {
    sycl::queue myQueue(sycl::gpu_selector_v);

    /* We need to create a buffer in order to access data
     * from the SYCL devices. */
    sycl::buffer<int, 1> buf(data, sycl::range<1>(nItems));

    /* This command group enqueues a kernel on myQueue
     * that adds the work-item id to each element of the
     * data array. Effectively, it creates an array of
     * consecutive integers. */
    myQueue.submit([&](sycl::handler& cgh) {
      auto ptr = buf.get_access<sycl::access::mode::read_write>(cgh);
      /* We create a linear (one dimensional) group range, which
       * creates a work-item per element of the vector. */
      auto groupRange = sycl::range<1>(nItems / nLocals);
      /* We create a linear (one dimensional) local range which defines the
       * workgroup size. */
      auto localRange = sycl::range<1>(nLocals);

      /* parallel_for_work_group takes a sycl::group as a parameter,
       * which has parallel_for_work_item as a method. */
      auto hierarchicalKernel = [=](sycl::group<1> groupID) {
        /* Unlike variables of any other type allocated in a
         * parallel_for_work_group scope, privateObj is allocated
         * per work-item and lives in work-item-private memory. */
	sycl::private_memory<int> privateObj(groupID);

        groupID.parallel_for_work_item([&](sycl::h_item<1> itemID) {
          /* Assign the work-item global id into private memory. */
          privateObj(itemID) = itemID.get_global_id()[0];
        });

        groupID.parallel_for_work_item([&](sycl::h_item<1> itemID) {
          /* Retrieve the global id stored in the previous
           * parallel_for_work_item call and store it in global memory. */
          auto globalID = privateObj(itemID);
          ptr[globalID] = globalID;
        });
      };
      cgh.parallel_for_work_group<PrivateMemory>(groupRange, localRange,
                                                 hierarchicalKernel);
    });
  }

  for (int i = 0; i < int(nItems); i++) {
    if (data[i] != i) {
      ret = 1;
    }
  }

  return ret;
}
