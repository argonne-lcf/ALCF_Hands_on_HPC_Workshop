# Intel Data Parallel Extension for Python

This folder collects environment setup and installation scripts for running Intel's DPEP packages on Aurora, along with some simple examples.
The material was created for the 2025 ALCF INCITE GPU Hackathon by Riccardo Balin.

Please reference the [hackathon slides](ALCF-Hackathon2025-DPEP.pdf) for more details or the [ALCF documentation](https://docs.alcf.anl.gov/aurora/data-science/python/#intels-data-parallel-extensions-for-python-dpep) page. 

## Environment Setup

To access the dpnp and dpctl packages, execute the setup script
```bash
source setup_env.sh
```

Numba-dpex is not included in the AI frameworks module, but it can be installed by either creating a new conda environment with 
```bash
source numba-dpex_install_new.sh
```

or by cloning the base environment with
```bash
source numba-dpex_install_clone.sh
``` 

Note that creating a new environment is quicker, however it will not contain all the packages present in the base module. If these packages are needed for your workload, users should clone the base environment.

## Examples

The following examples are included, divided by the respective packages

### dpnp
* [Sum on default device](examples/dpnp_sum.py)
* [Timing matmul kernel on device with sycl.queue.wait()](examples/dpnp_matmul.py)

### dpctl
* [Compute-follows-data](examples/compute_follows_data.py)
* [On-node device instrospection](examples/dpctl_device_introspection.py)
* [Device selection and SYCL queue creation](examples/dpctl_device_selection.py)

### numba-dpex
* [Sum range kernel](examples/dpex_sum.py)

### dlpack
* [PyTorch and dpnp interoperability](examples/dlpack_example.py)
 
