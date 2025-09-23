#include <sycl/sycl.hpp>
#include <iostream>

// Compile: clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80 sycl-multigpu.cpp -o sycl-multigpu.out

int main() {
    // Enumerate all GPUs visible to the job
    std::vector<sycl::device> gpus =
        sycl::device::get_devices(sycl::info::device_type::gpu);

    if (gpus.empty()) {
	std::cerr << "No GPUs visible to SYCL. Check your job/resources.\n";
	return 1;
    }

    std::cout << "Found " << gpus.size() << " GPU(s):\n";
    for (size_t i = 0; i < gpus.size(); ++i) {
	std::cout << "  [" << i << "] "
		  << gpus[i].get_info<sycl::info::device::name>()
		  << " (vendor: "
		  << gpus[i].get_info<sycl::info::device::vendor>() << ")\n";
    }

    // Create a queue per GPU and launch a tiny kernel on each
    std::vector<sycl::queue> queues;
    queues.reserve(gpus.size());
    for (auto& dev : gpus) {
	queues.emplace_back(dev); // per-device queue
    }

    // Launch: one single-task per device (so prints once per GPU)
    std::vector<sycl::event> events; events.reserve(queues.size());
    for (int i = 0; i < (int)queues.size(); ++i) {
	int idx = i;

	// Launch kernel on each GPU
	events.emplace_back(queues[i].single_task([=]() {
	    sycl::ext::oneapi::experimental::printf("Hello from GPU %d\n", idx);
	}));
    }

    // Ensure all kernels complete & prints flush
    for (auto& e : events) e.wait();
    for (auto& q : queues) q.wait_and_throw();

    std::cout << "All device kernels completed.\n";

  return 0;
}
