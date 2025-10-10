// SPDX-FileCopyrightText: Contributors to the HPCToolkit Project
//
// SPDX-License-Identifier: Apache-2.0

//==============================================================
// This code performs GPU offloading using sycl.
// =============================================================
#include "minitest.h"
#include <sycl/sycl.hpp>
#include <vector>
#include <iostream>
#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

// Use two vectors to manage multiple GPUs in our system. gpu_queues stores SYCL
// queues, with one queue for each GPU device. These queues allow us to submit
// work to specific GPUs. gpu_devices stores information about each GPU device,
// which can be useful for querying device properties.
std::vector<sycl::queue> gpu_queues;
std::vector<sycl::device> gpu_devices;

// Create an exception handler for asynchronous SYCL exceptions
static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
      std::cerr << "Exception in exception handler; Failure" << std::endl;
      std::terminate();
    }
  }
};

void
twork( int iter, int threadnum)
{
  hrtime_t iterstarttime = gethrtime();

  int nelements = nn;
  double *l1 = lptr[threadnum];
  double *r1 = rptr[threadnum];
  double *p1 = pptr[threadnum];
  int kernmax = kkmax;

  // Distribute work across GPU devices using round-robin assignment.
  // If MPI is used, consider both thread number and MPI rank; otherwise, use only thread number.
  int rank = 0;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  int gpu_index = (threadnum + rank) % gpu_queues.size();
  sycl::queue &q = gpu_queues[gpu_index];

  // Create buffers that hold the data shared between the host and the devices.
  // The buffer destructor is responsible to copy the data back to host when it
  // goes out of scope.
  sycl::buffer<double, 1> a(l1, nn);
  sycl::buffer<double, 1> b(r1, nn);
  sycl::buffer<double, 1> c(p1, nn);

  // Submit a command group to the queue by a lambda function that contains the
  // data access permission and device computation (kernel).
  q.submit([&](sycl::handler &h) {
    // Create an accessor for each buffer with access permission: read, write or
    // read/write. The accessor is a mean to access the memory in the buffer.
    sycl::accessor d_l1(a, h, sycl::read_only);
    sycl::accessor d_r1(b, h, sycl::read_only);
    sycl::accessor d_p1(c, h, sycl::read_write);

    // Use parallel_for to run vector addition in parallel on device. This
    // executes the kernel.
    //    1st parameter is the number of work items.
    //    2nd parameter is the kernel, a lambda that specifies what to do per
    //    work item. The parameter of the lambda is the work item id.
    // DPC++ supports unnamed lambda kernel by default.

    h.parallel_for(nelements, [=](auto i) {
#include "compute.h"
    } );
  }
  );
  q.wait();

  hrtime_t endtime = gethrtime();
  double  tempus =  (double) (endtime - iterstarttime) / (double)1000000000.;
  double  tempus2 =  (double) (endtime - starttime) / (double)1000000000.;
#if 1
 fprintf(stdout, "    [%d] Completed iteration %d, thread %d on GPU %d in %13.9f s. at timestamp %13.9f s.\n",
   thispid, iter, threadnum, gpu_index, tempus, tempus2);
#endif
  spacer(50, true);

}

void
initgpu()
{
  hrtime_t initstart = gethrtime();
  double  tempus =  (double) (initstart - starttime) / (double)1000000000.;
#if 1
 fprintf(stdout, "    [%d] Started initgpu() at timestamp %13.9f s.\n",
   thispid, tempus);
#endif
  try {

    // Discover all available GPU devices, which involves the following steps:
    // 1. Get all available SYCL platforms. (e.g., OpenCL, Level Zero)
    // 2. For each platform, get all GPU devices.
    // 3. For each GPU device:
    //    a. Add it to our list of devices (gpu_devices).
    //    b. Create a SYCL queue for it and add to gpu_queues.
    //    c. Print out the device information for logging purposes.
    std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
    for (const sycl::platform& platform : platforms) {
      std::vector<sycl::device> devices = platform.get_devices(sycl::info::device_type::gpu);
      for (const sycl::device& device : devices) {
        gpu_devices.push_back(device);
        gpu_queues.emplace_back(device, exception_handler);
        
        // Print out the device information used for the kernel code.
        std::cout << "    [" << thispid << "]" 
                  << " running on sycl platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << " with sycl device: "
                  << device.get_info<sycl::info::device::name>() << "\n";
      }
    }
  } catch (sycl::exception const &e) {
    std::cerr << "Failure: exception trying to determine sycl device(s).\n";
    std::terminate();
  }

  hrtime_t initdone = gethrtime();
  tempus =  (double) (initdone - starttime) / (double)1000000000.;
#if 1
 fprintf(stdout, "    [%d] Leaving initgpu() at timestamp %13.9f s.\n",
   thispid, tempus);
#endif
}
