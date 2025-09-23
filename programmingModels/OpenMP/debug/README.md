clang++ -fopenmp --offload-arch=native bug.cpp && ./a.out
segfault

Try on host
clang++ -fopenmp -fopenmp-targets=x86_64-pc-linux-gnu bug.cpp && ./a.out
Seems fine

using debugger
clang++ -fopenmp --offload-arch=native bug.cpp
cuda-gdb ./a.out
run

Thread 1 "a.out" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (96,0,0), device 0, sm 0, warp 3, lane 0]
0x0000555555edf970 in __omp_offloading_9491be2_4e000773_main_l9<<<(1,1,1),(128,1,1)>>> ()
Bad kernel


Add debug symbols
clang++ -fopenmp --offload-arch=native -g bug.cpp
cuda-gdb ./a.out
run

CUDA Exception: Warp Illegal Address
The exception was triggered at PC 0x555555ef1200 (bug.cpp:11)

Thread 1 "a.out" received signal CUDA_EXCEPTION_14, Warp Illegal Address.
[Switching focus to CUDA kernel 0, grid 1, block (0,0,0), thread (96,0,0), device 0, sm 0, warp 3, lane 0]
0x0000555555ef1210 in __omp_offloading_9491be2_4e000773_main_l9_debug__ (a=3, b_ptr=0x5555556144c0) at bug.cpp:11
11          printf("a=%d, b[1]=%d\n", a, b_ptr[1]);

Fix the transfers
  #pragma omp target map(to:b_ptr[:2])
