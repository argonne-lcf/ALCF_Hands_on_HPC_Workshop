# BERT (language model) is selected on hands-on section

Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google.


# SambaNova

1. Login to SN:  
```
ssh ALCFUserID@sambanova.alcf.anl.gov 
ssh sm-01 (or sm-02)
```

2. SDK setup:  
```
source /software/sambanova/envs/sn_env.sh
```

3. Copy scripts:  
```
cp /var/tmp/Additional/slurm/Models/ANL_Acceptance_RC1_11_5/bert_train-inf.sh ~/
```

4. Run scripts:  
```
cd ~
./bert_train-inf.sh
```


# Cerebras

1. Login to CS-2:  
```
ssh ALCFUserID@cerebras.alcf.anl.gov 
ssh cs2-01-med1
```

2. Copy scripts:  
```
cp -r /software/cerebras/model_zoo ~/  
cd model_zoo/modelzoo/transformers/tf/bert  
```
Ignore any permissions errors during the copy of the subdirectory `modelzoo-R1.3.0_2/`.


Next, modify `data_dir` to `'/software/cerebras/dataset/bert_large/msl128/'` in `configs/params_bert_large_msl128.yaml`, **in two places**.

4. Run scripts:  
```
MODELDIR=model_dir_bert_large_msl128_$(hostname)  
rm -r $MODELDIR  
time -p csrun_cpu python run.py --mode=train --compile_only --params configs/params_bert_large_msl128.yaml --model_dir $MODELDIR --cs_ip $CS_IP  
time -p csrun_wse python run.py --mode=train --params configs/params_bert_large_msl128.yaml --model_dir $MODELDIR --cs_ip $CS_IP
```

