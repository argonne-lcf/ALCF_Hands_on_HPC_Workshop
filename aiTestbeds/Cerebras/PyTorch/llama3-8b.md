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
2025-10-08 04:09:18,648 INFO:   ===========================================================================
2025-10-08 04:09:18,648 INFO:   Trainer Fit Summary
2025-10-08 04:09:18,648 INFO:   ---------------------------------------------------------------------------
2025-10-08 04:09:18,649 INFO:   Trainer will run 1 train loop, interleaving validation as follows:
2025-10-08 04:09:18,649 INFO:   * val_dataloader_2 after every train loop
2025-10-08 04:09:18,649 INFO:   
2025-10-08 04:09:18,649 INFO:   Train steps per train loop:
2025-10-08 04:09:18,649 INFO:   * 1 loop of 10 steps
2025-10-08 04:09:18,649 INFO:   for a total of 10 train steps.
2025-10-08 04:09:18,649 INFO:   
2025-10-08 04:09:18,649 INFO:   Checkpoints will be taken every 10 steps, for a total of 1 checkpoint.
2025-10-08 04:09:18,649 INFO:   
2025-10-08 04:09:18,649 INFO:   Progress will be logged every step.
2025-10-08 04:09:18,649 INFO:   ===========================================================================
2025-10-08 04:09:18,649 INFO:   ---------------------------------------------------------------------------
2025-10-08 04:09:18,649 INFO:   Starting train loop 1 of 1, from global step 1 to 10 (10 steps)
2025-10-08 04:09:18,649 INFO:   ---------------------------------------------------------------------------
2025-10-08 04:09:29,720 INFO:   Compiling the model. This may take a few minutes.
2025-10-08 04:09:31,869 INFO:   Initiating a new image build job against the cluster server.
2025-10-08 04:09:31,877 INFO:   User sidecar image build is disabled from server. Falling back to venv mounting.
2025-10-08 04:09:31,955 INFO:   Initiating a new compile wsjob against the cluster server.
2025-10-08 04:09:31,977 INFO:   Job id: wsjob-9tfhwpsvhxpcgfzt92rydm, workflow id: wflow-grjpwnj4xwr3jw3xtqvzyu, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-9tfhwpsvhxpcgfzt92rydm
2025-10-08 04:09:51,998 INFO:   Poll ingress status: Waiting for all Coordinator pods to be running, current running: 0/1.
2025-10-08 04:09:52,006 WARNING:   Event 2025-10-08 04:09:32 +0000 UTC reason=InconsistentVersion wsjob=wsjob-9tfhwpsvhxpcgfzt92rydm message='Warning: job image version 2.5.1-202507111115-6-48e76807 is inconsistent with cluster server version 3.0.1-202508200300-150-bba1322a+bba1322aed, there's a risk job could fail due to inconsistent setup.'
2025-10-08 04:10:02,016 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-9tfhwpsvhxpcgfzt92rydm&from=1759895980000&to=now
2025-10-08 04:10:02,019 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-9tfhwpsvhxpcgfzt92rydm&from=1759895980000&to=now
2025-10-08 04:10:02,825 INFO:   Found existing cached compile with hash: "cs_308109675380978773"
2025-10-08 04:10:06,645 INFO:   Compile artifacts successfully written to remote compile directory. Compile hash is: cs_308109675380978773
2025-10-08 04:10:12,029 INFO:   Compile was successful!
2025-10-08 04:10:12,030 INFO:   Waiting for weight initialization to complete
2025-10-08 04:10:12,030 INFO:   Programming Cerebras Wafer Scale Cluster for execution. This may take a few minutes.
2025-10-08 04:10:14,112 INFO:   Initiating a new execute wsjob against the cluster server.
2025-10-08 04:10:14,143 INFO:   Job id: wsjob-vk5ai9fnrbkec2bm56tt28, workflow id: wflow-grjpwnj4xwr3jw3xtqvzyu, namespace: job-operator, remote log path: /n1/wsjob/workdir/job-operator/wsjob-vk5ai9fnrbkec2bm56tt28
2025-10-08 04:10:34,164 INFO:   Poll ingress status: Waiting for all Activation pods to be running, current running: 0/3.
2025-10-08 04:10:34,176 WARNING:   Event 2025-10-08 04:10:15 +0000 UTC reason=InconsistentVersion wsjob=wsjob-vk5ai9fnrbkec2bm56tt28 message='Warning: job image version 2.5.1-202507111115-6-48e76807 is inconsistent with cluster server version 3.0.1-202508200300-150-bba1322a+bba1322aed, there's a risk job could fail due to inconsistent setup.'
2025-10-08 04:11:44,246 INFO:   Poll ingress status: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-vk5ai9fnrbkec2bm56tt28&from=1759896023000&to=now
2025-10-08 04:11:44,254 INFO:   Poll ingress success: Job ingress ready, dashboard: https://grafana.anl0.cerebras.internal/d/WebHNShVz/wsjob-dashboard?orgId=1&var-wsjob=wsjob-vk5ai9fnrbkec2bm56tt28&from=1759896023000&to=now
2025-10-08 04:11:44,331 INFO:   Preparing to execute using 1 CSX
2025-10-08 04:12:24,445 INFO:   About to send initial weights
2025-10-08 04:12:42,888 INFO:   Finished sending initial weights
2025-10-08 04:12:42,889 INFO:   Finalizing appliance staging for the run
2025-10-08 04:12:42,896 INFO:   Waiting for device programming to complete
2025-10-08 04:16:34,735 INFO:   Device programming is complete
2025-10-08 04:16:35,848 INFO:   Using network type: ROCE
2025-10-08 04:16:35,849 INFO:   Waiting for input workers to prime the data pipeline and begin streaming ...
2025-10-08 04:16:35,854 INFO:   Input workers have begun streaming input data
2025-10-08 04:16:36,976 INFO:   Appliance staging is complete
2025-10-08 04:16:36,976 INFO:   Beginning appliance run
2025-10-08 04:17:27,583 INFO:   | Train Device=CSX, Step=1, Loss=12.59069, Rate=18.99 samples/sec, GlobalRate=18.99 samples/sec, LoopTimeRemaining=0:07:35, TimeRemaining>0:07:35
2025-10-08 04:18:17,001 INFO:   | Train Device=CSX, Step=2, Loss=12.59906, Rate=19.25 samples/sec, GlobalRate=19.20 samples/sec, LoopTimeRemaining=0:06:40, TimeRemaining>0:06:40
2025-10-08 04:19:05,792 INFO:   | Train Device=CSX, Step=3, Loss=12.60336, Rate=19.51 samples/sec, GlobalRate=19.36 samples/sec, LoopTimeRemaining=0:05:46, TimeRemaining>0:05:46
2025-10-08 04:19:55,349 INFO:   | Train Device=CSX, Step=4, Loss=12.60226, Rate=19.43 samples/sec, GlobalRate=19.36 samples/sec, LoopTimeRemaining=0:04:57, TimeRemaining>0:04:57
2025-10-08 04:20:44,794 INFO:   | Train Device=CSX, Step=5, Loss=12.59109, Rate=19.42 samples/sec, GlobalRate=19.37 samples/sec, LoopTimeRemaining=0:04:07, TimeRemaining>0:04:07
2025-10-08 04:21:34,150 INFO:   | Train Device=CS
  ```
</details>
