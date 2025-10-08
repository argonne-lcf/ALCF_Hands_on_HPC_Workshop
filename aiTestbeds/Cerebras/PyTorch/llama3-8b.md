# LLAMA3-8B on Cerebras

##### Go to directory with llama3-8b example. 
```bash
mkdir llama_38b
```

#####  Activate PyTorch virtual Environment 
```bash
source ~/R_2.5.0/venv_cerebras_pt/bin/activate
```

#####  Replace config file with correct configurations file. 
```bash
cp /software/cerebras/dataset/params_llama3_8b_msl_2k.yaml .
```

#####  Run Training Job
```bash
cszoo fit params_llama3_8b_msl_2k.yaml --disable_version_check 2>&1 | tee ~/out_llama3_8b.log
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
  ```
</details>
