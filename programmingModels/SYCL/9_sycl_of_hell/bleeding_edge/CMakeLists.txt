add_exe(4_buffer_id_introspection)
add_exe(omp_sycl)

##does not compile: reduction.hpp:1816:10: error: no matching constructor for initialization of ...
add_exe(reduction) 

##does not compile: vadd_usm.cpp:53:36: error: expected namespace name
#add_exe(vadd_usm)
##warning: the flag '-fsycl-explicit-simd' has been deprecated and will be ignored
#target_compile_options(vadd_usm PRIVATE -fsycl-explicit-simd)

##does not link: undefined reference to `zeDeviceGetProperties`
#add_exe(ze_interop) 
