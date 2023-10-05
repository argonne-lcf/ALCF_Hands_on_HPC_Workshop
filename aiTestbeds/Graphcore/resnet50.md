# Resnet50 on Graphcore 

Go to direcotry with Resnet50 example 
```bash
cd ~/graphcore/examples/vision/cnns/pytorch
```

Create a new PopTorch Environment 
```bash
POPLAR_SDK_ROOT=/software/graphcore/poplar_sdk/3.3.0/
export POPLAR_SDK_ROOT=$POPLAR_SDK_ROOT

virtualenv ~/Graphcore/workspace/poptorch33_resnet50_env
source ~/Graphcore/workspace/poptorch33_resnet50_env/bin/activate
pip install $POPLAR_SDK_ROOT/poptorch-3.3.0+113432_960e9c294b_ubuntu_20_04-cp38-cp38-linux_x86_64.whl
export PYTHONPATH=$POPLAR_SDK_ROOT/python:$PYTHONPATH
```

Install Requirements 

```bash
cd ~/graphcore/examples/vision/cnns/pytorch
make install 
make install-turbojpeg

python -m pip install -r requirements.txt
```

Update Configuration

Change following line in `configs.yml`
```yaml
use_bbox_info: false
```

Run Resnet50

Create a script called `poprun_resnet.sh` with following 
```bash
#!/bin/bash
poprun -vv --vipu-partition=slurm_${SLURM_JOBID} --num-instances=1 --num-replicas=4 --executable-cache-path=$PYTORCH_CACHE_DIR python3 /home/$USER/graphcore/examples/vision/cnns/pytorch/train/train.py --config resnet50-pod4 --imagenet-data-path /mnt/localdata/datasets/imagenet-raw-dataset --epoch 2 --validation-mode none --dataloader-worker 14 --dataloader-rebatch-size 256
```

submit a job
```bash
chmod +x poprun_resnet.sh
/opt/slurm/bin/srun --ipus=4 poprun_resnet.sh
```
<details>
  <summary>Sample Output</summary>
  
  ```bash
    srun: job 10675 queued and waiting for resources
    srun: job 10675 has been allocated resources
    23:48:29.160 3555537 POPRUN [I] V-IPU server address picked up from 'vipu': 10.1.3.101:8090
    23:48:29.160 3555537 POPRUN [D] Connecting to 10.1.3.101:8090
    23:48:29.162 3555537 POPRUN [D] Status for partition slurm_10673: OK (error 0)
    23:48:29.162 3555537 POPRUN [I] Partition slurm_10673 already exists and is in state: PS_ACTIVE
    23:48:29.163 3555537 POPRUN [D] The reconfigurable partition slurm_10673 is OK
    ===========================
    |      poprun topology      |
    |===========================|
    | hosts     | gc-poplar-02  |
    |-----------|---------------|
    | ILDs      |       0       |
    |-----------|---------------|
    | instances |       0       |
    |-----------|---------------|
    | replicas  | 0 | 1 | 2 | 3 |
    ---------------------------
    23:48:29.163 3555537 POPRUN [D] Target options from environment: {}
    23:48:29.163 3555537 POPRUN [D] Target options from V-IPU partition: {"ipuLinkDomainSize":"4","ipuLinkConfiguration":"default","ipuLinkTopology":"mesh","gatewayMode":"true","instanceSize":"4"}
    23:48:29.207 3555537 POPRUN [D] Found 1 devices with 4 IPUs
    23:48:29.777 3555537 POPRUN [D] Attached to device 6
    23:48:29.777 3555537 POPRUN [I] Preparing parent device 6
    23:48:29.777 3555537 POPRUN [D] Device 6 ipuLinkDomainSize=64, ipuLinkConfiguration=Default, ipuLinkTopology=Mesh, gatewayMode=true, instanceSize=4
    23:48:33.631 3555537 POPRUN [D] Target options from Poplar device: {"ipuLinkDomainSize":"64","ipuLinkConfiguration":"default","ipuLinkTopology":"mesh","gatewayMode":"true","instanceSize":"4"}
    23:48:33.631 3555537 POPRUN [D] Using target options: {"ipuLinkDomainSize":"4","ipuLinkConfiguration":"default","ipuLinkTopology":"mesh","gatewayMode":"true","instanceSize":"4"}


    Graph compilation: 100%|██████████| 100/100 [00:04<00:00][1,0]<stderr>:2023-08-22T23:49:40.103248Z PO:ENGINE   3556102.3556102 W: WARNING: The compile time engine option debug.branchRecordTile is set to "5887" when creating the Engine. (At compile time it was set to 1471)
    [1,0]<stderr>:
    Loss:6.7539 [1,0]<stdout>:[INFO] Epoch 1████▌| 75/78 [02:42<00:06,  2.05s/it][1,0]<stderr>:
    [1,0]<stdout>:[INFO] loss: 6.7462,
    [1,0]<stdout>:[INFO] accuracy: 0.62 %
    [1,0]<stdout>:[INFO] throughput: 7599.7 samples/sec
    [1,0]<stdout>:[INFO] Epoch 2/2
    Loss:6.7462 | Accuracy:0.62%: 100%|██████████| 78/78 [02:48<00:00,  2.16s/it][1,0]<stderr>:
    Loss:6.2821 | Accuracy:2.42%:  96%|█████████▌| 75/7[1,0]<stdout>:[INFO] Epoch 2,0]<stderr>:
    [1,0]<stdout>:[INFO] loss: 6.2720,
    [1,0]<stdout>:[INFO] accuracy: 2.48 %
    [1,0]<stdout>:[INFO] throughput: 8125.8 samples/sec
    [1,0]<stdout>:[INFO] Finished training. Time: 2023-08-22 23:54:57.853508. It took: 0:05:26.090631
    Loss:6.2720 | Accuracy:2.48%: 100%|██████████| 78/78 [02:37<00:00,  2.02s/it][1,0]<stderr>:
    [1,0]<stderr>:/usr/lib/python3.8/multiprocessing/resource_tracker.py:216: UserWarning: resource_tracker: There appear to be 14 leaked semaphore objects to clean up at shutdown
    [1,0]<stderr>:  warnings.warn('resource_tracker: There appear to be %d '
    23:55:02.722 3555537 POPRUN [I] mpirun (PID 3556098) terminated with exit code 0
  ```
</details>