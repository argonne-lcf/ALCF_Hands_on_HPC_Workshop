# Standard Language Parallelism

## Fortran

In this example we are solving a linear system Ax = B where A is square. This test will run on
the CPU or GPU, depending on the compilation/linking options.

To set up the environment, make sure you have the NVIDIA HPC SDK loaded.

### Exercise 1

Verify you can compile the code to run on the CPU.

```
nvfortran -o testdgetrf_cpu testdgetrf.F90 -lblas # or other BLAS library
```


### Exercise 2

Recompile the code to target the GPU.

```
nvfortran -o testdgetrf_gpu testdgetrf.F90 -stdpar -cuda -gpu=nvlamath,cuda11.4 -cudalib=nvlamath,curand
```

Note that because we use the `-stdpar` option in the GPU build, all Fortran allocatable arrays
use managed memory, and can be used on either the CPU or GPU.

Now use the GPU build instead of the CPU build. Again verify the test passes.

### Exercise 3

Collect a profile of the GPU application using Nsight Systems. Use `nsys profile --stats=true` then the executable name. Verify that the GPU build actually executed a GPU kernel.

### Exercise 4

Vary `M` and `N` (keeping the matrix square for simplicity). Collect a profile for each case and see how the
performance depends on the linear system size. You may want to use the `-o` option to Nsight Systems to name
your profiles explicitly (e.g. `nsys profile --stats=true -o dgetrf_1000`).

## C++

We will look at a saxpy example using standard C++.

## Exercise 1

Compile for the GPU:
```
nvc++ -stdpar=gpu -o saxpy_gpu ./saxpy.cpp
```

(The `-stdpar` default is `gpu` so you do not need to explicitly use that option, but it is useful to
be explicit, especially when switching between CPU and GPU platforms.)

Run the code, using the appropriate script/commandline for the site you're running at:

You should see SUCCESS written to the job output file or command line if everything worked.

## Exercise 2

Profile the code with Nsight Systems to verify it actually ran on the GPU.

## Exercise 3

Compile the process for CPU parallelism using the `-stdpar=multicore` command.
```
nvc++ -stdpar=multicore -o saxpy_cpu ./saxpy.cpp
```

Run with the new executable and verify correct execution.
