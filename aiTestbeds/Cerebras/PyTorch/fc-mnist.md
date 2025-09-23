# FC-MNIST On Cerebras

A simple multi-layer perceptron model composed of fully-connected layers for performing handwriting recognition on the MNIST dataset.


Go to directory with fc_mnist example. 
```bash
cd ~/R_2.3.0/modelzoo/modelzoo/fc_mnist/pytorch
```

Activate PyTroch virtual Environment 
```bash
source ~/R_2.3.0/venv_pt/bin/activate
```

Edit configs/params.yaml, making the following changes:
```yaml
 train_input:
-    data_dir: "./data/mnist/train"
+    data_dir: "/software/cerebras/dataset/fc_mnist/data/mnist/train"

 eval_input:
-    data_dir: "./data/mnist/val"
+    data_dir: "/software/cerebras/dataset/fc_mnist/data/mnist/train"
```

Run Training Job

Following commands Compile and Execute models. 
```bash
export MODEL_DIR=model_dir
# deletion of the model_dir is only needed if sample has been previously run
if [ -d "$MODEL_DIR" ]; then rm -Rf $MODEL_DIR; fi
python run.py CSX --job_labels name=pt_smoketest --params configs/params.yaml --num_csx=1 --mode train --model_dir $MODEL_DIR --mount_dirs /home/ /software --python_paths /home/$(whoami)/R_2.3.0/modelzoo --compile_dir /$(whoami) |& tee mytest.log
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
    2023-05-15 16:05:54,510 INFO:   | Train Device=xla:0, Step=9950, Loss=2.30234, Rate=157300.30 samples/sec, GlobalRate=26805.42 samples/sec
    2023-05-15 16:05:54,571 INFO:   | Train Device=xla:0, Step=10000, Loss=2.29427, Rate=125599.14 samples/sec, GlobalRate=26905.42 samples/sec
    2023-05-15 16:05:54,572 INFO:   Saving checkpoint at global step 10000
    2023-05-15 16:05:59,734 INFO:   Saving step 10000 in dataloader checkpoint
    2023-05-15 16:06:00,117 INFO:   Saved checkpoint at global step: 10000
    2023-05-15 16:06:00,117 INFO:   Training Complete. Completed 1280000 sample(s) in 53.11996841430664 seconds.
    2023-05-15 16:06:04,356 INFO:   Monitoring returned
  ```
</details>