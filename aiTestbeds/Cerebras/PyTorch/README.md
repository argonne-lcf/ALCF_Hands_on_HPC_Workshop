# Cerebras ML Software Stack



## Create Virtual Environment 


```bash
#Make your home directory navigable
mkdir ~/R_2.3.0
cd ~/R_2.3.0
# Note: "deactivate" does not actually work in scripts.
deactivate
rm -r venv_cerebras_pt
/software/cerebras/python3.8/bin/python3.8 -m venv venv_cerebras_pt
source venv_cerebras_pt/bin/activate
pip install --upgrade pip
pip install cerebras_pytorch==2.3.0
```

## Clone Cerebras modelzoo

We use example from [Cerebras Modelzoo repository](https://github.com/Cerebras/modelzoo) for this hands-on. 

* Clone the modezoo repository.

    ```bash
    mkdir ~/R_2.3.0
    cd ~/R_2.3.0
    git clone https://github.com/Cerebras/modelzoo.git
    cd modelzoo
    git tag
    git checkout Release_2.3.0
    ```

* Activate PyTroch virtual Environment 
    ```bash
    source ~/R_2.2.1/venv_cerebras_pt/bin/activate
    pip install -r ~/R_2.2.1/modelzoo/requirements.txt
    ```

## Hands-on Session Example

* [LLAMA2-7B](./llama2-7b.md)


## Next Steps

Try additonal examples. 
* [FC-MNIST](./fc-mnist.md)
* [BERT-Large](./bert-large.md)
  

# Useful Resources 

* [Cerebras Documntation](https://docs.cerebras.net/en/latest/wsc/index.html)
* [Cerebras Modelzoo Repo](https://github.com/Cerebras/modelzoo/tree/main/modelzoo)
* Datasets Path: `/software/cerebras/dataset`