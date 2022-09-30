# Profiling TensorFlow

In this example, we'll profile a Generative network.  We'll go through several steps of profile, each time enabling a new tool or optimization.  At the end of the exercise, you'll have a GAN that can generate images like this:

![Images](images/generated_images.png)

The tools behind the GAN are not the subject for this talk but feel free to reach out on Slack if you have questions about it!

Find the original script in `train_GAN.py`.

All the scripts used here work in the conda module:

```bash
$ module load conda/2022-09-08
$ conda activate
```


## A Starting Point

<!-- 
To download the mnist dataset, make sure to enable http forwarding:
```bash
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128
``` -->

Run the original script, single node, like so: `python train_GAN.py`.  Feel free to ctrl+C once it hits a stable throughput.

Take note of the throughput reported!

```
2021-04-30 20:21:57,881 - INFO - (0, 0), G Loss: 0.718, D Loss: 0.694, step_time: 2.402, throughput: 53.286 img/s.
2021-04-30 20:21:58,567 - INFO - (0, 1), G Loss: 0.831, D Loss: 0.636, step_time: 0.686, throughput: 186.688 img/s.
2021-04-30 20:21:59,222 - INFO - (0, 2), G Loss: 0.855, D Loss: 0.608, step_time: 0.655, throughput: 195.538 img/s.
2021-04-30 20:21:59,801 - INFO - (0, 3), G Loss: 0.859, D Loss: 0.592, step_time: 0.578, throughput: 221.365 img/s.
2021-04-30 20:22:00,362 - INFO - (0, 4), G Loss: 0.814, D Loss: 0.599, step_time: 0.561, throughput: 228.349 img/s.
2021-04-30 20:22:00,902 - INFO - (0, 5), G Loss: 0.770, D Loss: 0.593, step_time: 0.540, throughput: 236.824 img/s.
2021-04-30 20:22:01,443 - INFO - (0, 6), G Loss: 0.727, D Loss: 0.600, step_time: 0.540, throughput: 236.819 img/s.
2021-04-30 20:22:01,983 - INFO - (0, 7), G Loss: 0.707, D Loss: 0.597, step_time: 0.540, throughput: 237.101 img/s.
2021-04-30 20:22:02,523 - INFO - (0, 8), G Loss: 0.710, D Loss: 0.593, step_time: 0.540, throughput: 237.136 img/s.
2021-04-30 20:22:03,063 - INFO - (0, 9), G Loss: 0.678, D Loss: 0.610, step_time: 0.540, throughput: 237.176 img/s.
2021-04-30 20:22:03,619 - INFO - (0, 10), G Loss: 0.676, D Loss: 0.609, step_time: 0.540, throughput: 237.097 img/s.
2021-04-30 20:22:04,159 - INFO - (0, 11), G Loss: 0.665, D Loss: 0.616, step_time: 0.540, throughput: 237.068 img/s
```

On average, the A100 system is moving about 237 Images / second through this training loop.  Let's dig in to the first optimization in the [`line_profiler`](https://github.com/argonne-lcf/CompPerfWorkshop-2021/tree/main/09_profiling_frameworks/TensorFlow/line_profiler) directory.

Below are the wrap up conclusions which you can read ahead or come back to later.

# Conclusions

Try the `optimized` version of the code - what throughput are you getting?  It should be a good deal faster! (~170000 Img/s - about 700x faster)  So, after all the profiling, what optimizations did we learn?

 - Make sure that IO isn't a bottleneck.  In this case it was simple.  With big datasets it can be a challenge to keep the GPU fed and not idle on IO.
 - Make sure to use graph compilation where you can.  It's easy to make mistakes here: you must make sure to use only tensorflow operations!
 - Use XLA.  It can give excellent speed ups by fusing operations.
 - Use reduced precision ... if there isn't a bug.  Reduced precision becomes particularly powerful when XLA is involved, allowing you to keep the tensorcores chugging along with less memory-bound operations.

In general, if you have an application running in tensorflow, it's a great idea to profile periodically and make sure you've got all the basic optimizations down!
