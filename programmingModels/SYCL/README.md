Queue (PBS one, not SYCL... OhOhOh):
```
qsub -I -l select=1,walltime=0:60:00,filesystems=home:grand -A alcf_training -q HandsOnHPC
```

Variable to set to compile
```
# Polaris
module use /soft/modulefiles
module load oneapi/upstream # Can do module `show` to get more help. Or `help`
export CXX=clang++ 
export  CXXFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80" 
make -j
```
