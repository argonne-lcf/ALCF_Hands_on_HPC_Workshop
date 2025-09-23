# BERT-Large on Cerebras

Go to directory with fc_mnist example. 
```bash
cd ~/R_2.3.0/modelzoo/modelzoo/transformers/pytorch/bert
```

Activate PyTroch virtual Environment 
```bash
source ~/R_2.3.0/venv_pt/bin/activate
```

Replace config file with correct configurations file. 
```bash
cp /software/cerebras/dataset/bert_large/bert_large_MSL128_sampleds.yaml configs/bert_large_MSL128_sampleds.yaml
```

Run Training Job
```bash
export MODEL_DIR=model_dir_bert_large_pytorch
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
python run.py CSX --job_labels name=bert_pt --params configs/bert_large_MSL128_sampleds.yaml --num_workers_per_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software/ --python_paths /home/$(whoami)/R_2.3.0/modelzoo/ --compile_dir $(whoami) |& tee mytest.log
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
    2023-05-17 18:10:08,776 INFO:   Finished sending initial weights
    2023-05-17 18:15:11,548 INFO:   | Train Device=xla:0, Step=100, Loss=9.46875, Rate=4597.49 samples/sec, GlobalRate=4597.49 samples/sec
    2023-05-17 18:15:23,067 INFO:   | Train Device=xla:0, Step=200, Loss=8.94531, Rate=7173.00 samples/sec, GlobalRate=6060.68 samples/sec
    2023-05-17 18:15:41,547 INFO:   | Train Device=xla:0, Step=300, Loss=8.79688, Rate=6193.85 samples/sec, GlobalRate=5876.98 samples/sec
    2023-05-17 18:15:54,118 INFO:   | Train Device=xla:0, Step=400, Loss=8.28906, Rate=7365.06 samples/sec, GlobalRate=6316.84 samples/sec
    2023-05-17 18:16:12,430 INFO:   | Train Device=xla:0, Step=500, Loss=8.14844, Rate=6301.21 samples/sec, GlobalRate=6157.22 samples/sec
    2023-05-17 18:16:25,177 INFO:   | Train Device=xla:0, Step=600, Loss=8.06250, Rate=7340.44 samples/sec, GlobalRate=6406.58 samples/sec
    2023-05-17 18:16:43,315 INFO:   | Train Device=xla:0, Step=700, Loss=8.00000, Rate=6323.57 samples/sec, GlobalRate=6285.55 samples/sec
    2023-05-17 18:16:56,110 INFO:   | Train Device=xla:0, Step=800, Loss=7.96484, Rate=7331.29 samples/sec, GlobalRate=6458.82 samples/sec
    2023-05-17 18:17:14,564 INFO:   | Train Device=xla:0, Step=900, Loss=7.89844, Rate=6261.77 samples/sec, GlobalRate=6343.22 samples/sec
    2023-05-17 18:17:26,977 INFO:   | Train Device=xla:0, Step=1000, Loss=7.90234, Rate=7454.38 samples/sec, GlobalRate=6493.27 samples/sec
    2023-05-17 18:17:26,978 INFO:   Saving checkpoint at global step 1000
    2023-05-17 18:18:38,485 INFO:   Saving step 1000 in dataloader checkpoint
    2023-05-17 18:18:38,931 INFO:   Saved checkpoint at global step: 1000
    2023-05-17 18:18:38,932 INFO:   Training Complete. Completed 1024000 sample(s) in 229.65675950050354 seconds.
    2023-05-17 18:18:49,293 INFO:   Monitoring returned
  ```
</details>