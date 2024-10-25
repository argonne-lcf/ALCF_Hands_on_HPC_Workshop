vadd_usm: vadd_usm.cpp
	dpcpp vadd_usm.cpp -fsycl-explicit-simd -o vadd_usm

run_vadd_usm:
	SYCL_PROGRAM_COMPILE_OPTIONS=-vc-codegen ./vadd_usm

omp_syc:
	icpx -fiopenmp -fopenmp-targets=spir64 -fsycl omp_sycl.cpp

run_comp_syc:
	./run_comp_syc

4_buffer_id_introspection: 4_buffer_id_introspection.cpp
	dpcpp 4_buffer_id_introspection.cpp -o 4_buffer_id_introspection

run_4_buffer_id_introspection:
	./4_buffer_id_introspection.cpp

clean:
	rm -rf vadd_usm omp_syc
