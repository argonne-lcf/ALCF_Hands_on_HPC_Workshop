# Getting Started with Poplar

## SDK Overview 



There are two ways to run examples. 
1. IPU model (Simulator) 
2. On IPU Hardware

## Setup Poplar SDK

* Poplar SDK should be enabled by default, if its not, enable it.
    ```bash
    > source /software/graphcore/poplar_sdk/3.3.0/enable
    > popc --version
    POPLAR version 3.3.0 (de1f8de2a7)
    clang version 16.0.0 (2fce0648f3c328b23a6cbc664fc0dd0630122212)
    ```

* Go to directory with GEMV code. 
  ```bash
  cd examples/tutorials/tutorials/poplar/tut5_matrix_vector/complete
  ```
## Run with IPU Model

* Compile `tut5_complete.cpp` with the provided `Makefile`
    ```bash
    make tut5
    ```
* Run executable on the CPU. 
    ```bash
    ./tut5 1000 100
    ``` 

    <details>
    <summary>Sample Output</summary>
    
    ```bash
        ./tut5 1000 100
        Multiplying matrix of size 1000x100 by vector of size 100
        Creating new graph object and compiling vertex program additions
        Constructing full compute graph and control program
        Running graph program to multiply matrix by vector
        Multiplication result OK
    ```
    </details>

## Run on IPU

* Compile `tut5_ipu_hardware_complete.cpp` with the provided `Makefile`
    ```bash
    make tut5_ipu
    ```

* Run executable on IPU using scheduler.
    ```bash
    srun --ipus=1 ./tut5_ipu 10000 1000 --device ipu
    ```
    <details>
    <summary>Sample Output</summary>
    
    ```bash
        srun --ipus=1 ./tut5_ipu 1000 100
        srun: job 26636 queued and waiting for resources
        srun: job 26636 has been allocated resources
        Multiplying matrix of size 1000x100 by vector of size 100
        Trying to attach to IPU
        Attached to IPU 0
        Creating environment (compiling vertex programs)
        Constructing compute graph and control program
        Running graph program to multiply matrix by vector
        Multiplication result OK
    ```
    </details>




## Next Steps

Follow [Poplar Tutorial](https://github.com/graphcore/examples/tree/master/tutorials/tutorials/poplar) for optimized implementation of GEMV.

## Useful Resources

* [Poplar and PopLibs User Guide](https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/)
* [Poplar and PopLibs API Reference](https://docs.graphcore.ai/projects/poplar-api/en/latest/)
* [Request Cerebras SDK](https://cerebras.ai/homepage-landing/developers/sdk-request/)




