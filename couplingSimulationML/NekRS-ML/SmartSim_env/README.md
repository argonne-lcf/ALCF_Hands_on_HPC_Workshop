# Build the environment to run SmartSim

1. Get an interactive node with
```
./subInteractive.sh
```
2. Copy the build script `build_SSIM_Polaris.sh` to the location where you wish to place the new Conda environment
3. Run the build script as 
```
source build_SSIM_Polaris.sh
```
4. Apply the patch to the SmartSim source code outlined in [this PR](https://github.com/CrayLabs/SmartSim/pull/282)
