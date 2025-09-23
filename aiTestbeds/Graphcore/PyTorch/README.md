# Graphcore PyTorch


## Create Virtual Environment 

If the Poplar SDK is not enabled, it can be enabled with
```bash
source /software/graphcore/poplar_sdk/3.3.0/enable
```

### PyTorch virtual environment

```bash
mkdir -p ~/venvs/graphcore
virtualenv ~/venvs/graphcore/poptorch33_env
source ~/venvs/graphcore/poptorch33_env/bin/activate

POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.3.0
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT
pip install $POPLAR_SDK_ROOT/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
```


## Run Examples

Refer to respective instrcutions below 

* [GPT2](./gpt2.md)


## Next Steps 

Try additonal examples in this repo. 
* [MNIST](./mnist.md)
* [Resnet50 using replication factor](./resnet50.md)
* Go through tutorials and examples from [Graphcore Examples Repository](https://github.com/graphcore/examples)

## Useful Resources 

* [Graphcore Documentation](https://docs.graphcore.ai/en/latest/)
* [Graphcore Examples Repository](https://github.com/graphcore/examples)
* Graphcore SDK Path: `/software/graphcore/poplar_sdk`
