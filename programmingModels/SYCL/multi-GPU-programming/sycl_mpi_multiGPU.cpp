#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <mpi.h>
#include <sched.h>
#include <sycl/sycl.hpp>
#include <omp.h>

// SYCL port of https://code.ornl.gov/olcf/hello_jobstep
// To compile: mpicxx -fsycl -fopenmp -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 sycl_mpi_multiGPU.cpp  -o sycl_mpi_multiGPU.out
// To run: mpiexec -n 4 --ppn 4 --env OMP_NUM_THREADS=1 ./set_polaris_affinity.sh ./sycl_mpi_multiGPU.out

int main(int argc, char *argv[]){

  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  char name[MPI_MAX_PROCESSOR_NAME];
  int resultlength;
  MPI_Get_processor_name(name, &resultlength);

  // If CUDA_VISIBLE_DEVICES is set, capture visible GPUs
  const char* gpu_id_list;
  const char* cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
  if(cuda_visible_devices == NULL){
    gpu_id_list = "N/A";
  }
  else{
    gpu_id_list = cuda_visible_devices;
  }

  // Find how many GPUs L0 runtime says are available
  int num_devices = 0;
  std::vector<sycl::device> sycl_all_devs = sycl::device::get_devices(sycl::info::device_type::gpu);
  num_devices = sycl_all_devs.size();

  int hwthread;
  int thread_id = 0;

  if(num_devices == 0){
#pragma omp parallel default(shared) private(hwthread, thread_id)
    {
      thread_id = omp_get_thread_num();
      hwthread = sched_getcpu();

      printf("MPI %03d - OMP %03d - HWT %03d - Node %s\n",
             rank, thread_id, hwthread, name);

    }
  }
  else{

    std::string busid = "";

    std::string busid_list = "";
    std::string rt_gpu_id_list = "";

    // Loop over the GPUs available to each MPI rank
    for(int i=0; i<num_devices; i++){

      // // Get the PCIBusId for each GPU and use it to query for UUID
      busid = sycl_all_devs[i].get_info<sycl::ext::intel::info::device::pci_address>();
      busid_list.append(busid);

      // Concatenate per-MPIrank GPU info into strings for print
      if(i > 0) rt_gpu_id_list.append(",");
      rt_gpu_id_list.append(std::to_string(i));
    }

#pragma omp parallel default(shared) private(hwthread, thread_id)
    {
#pragma omp critical
      {
        thread_id = omp_get_thread_num();
        hwthread = sched_getcpu();

        printf("MPI %03d - OMP %03d - HWT %03d - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s\n",
               rank, thread_id, hwthread, name, rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
      }
    }
  }

  MPI_Finalize();

  return 0;
}
