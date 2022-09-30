# TF Function and Graph Compilation

Using line profiler showed us that the largest computation use, by far, was the train loop and is subcalls.  Here, we'll wrap those functions in `@tf.function` decorators to improve performance with graph compilation.  This has a mild overhead at the start for tracing but then runs more quickly.  Note the first few iterations are slower.

Out of the box, we see a dramatic speed up:

```2021-04-28 20:43:55,813 - INFO - G Loss: 0.786, D Loss: 0.619, step_time: 0.007, throughput: 18357.699 img/s.
2021-04-28 20:43:55,820 - INFO - G Loss: 0.688, D Loss: 0.620, step_time: 0.007, throughput: 18342.646 img/s.
2021-04-28 20:43:55,827 - INFO - G Loss: 0.806, D Loss: 0.622, step_time: 0.007, throughput: 18377.808 img/s.
2021-04-28 20:43:55,834 - INFO - G Loss: 0.673, D Loss: 0.624, step_time: 0.007, throughput: 18433.336 img/s.
2021-04-28 20:43:55,841 - INFO - G Loss: 0.816, D Loss: 0.623, step_time: 0.007, throughput: 18409.317 img/s.
2021-04-28 20:43:55,848 - INFO - G Loss: 0.677, D Loss: 0.623, step_time: 0.007, throughput: 18393.549 img/s.
2021-04-28 20:43:55,855 - INFO - G Loss: 0.802, D Loss: 0.622, step_time: 0.007, throughput: 18358.327 img/s.
2021-04-28 20:43:55,862 - INFO - G Loss: 0.687, D Loss: 0.621, step_time: 0.007, throughput: 18391.029 img/s.
2021-04-28 20:43:55,870 - INFO - G Loss: 0.784, D Loss: 0.620, step_time: 0.007, throughput: 18428.274 img/s.
2021-04-28 20:43:55,877 - INFO - G Loss: 0.695, D Loss: 0.621, step_time: 0.007, throughput: 18387.879 img/s.
2021-04-28 20:43:55,884 - INFO - G Loss: 0.783, D Loss: 0.620, step_time: 0.007, throughput: 18403.007 img/s.
2021-04-28 20:43:55,891 - INFO - G Loss: 0.695, D Loss: 0.621, step_time: 0.007, throughput: 18380.325 img/s.
2021-04-28 20:43:55,898 - INFO - G Loss: 0.791, D Loss: 0.622, step_time: 0.007, throughput: 18369.005 img/s.
2021-04-28 20:43:55,905 - INFO - G Loss: 0.687, D Loss: 0.622, step_time: 0.007, throughput: 18463.131 img/s.
2021-04-28 20:43:55,912 - INFO - G Loss: 0.798, D Loss: 0.622, step_time: 0.007, throughput: 18416.263 img/s.
```

Instead of 230 Img/s, we're at 18000 Img/s.  But note that the step time is now only 0.007 seconds.  We're likely not doing enough work per batch.  Change it to a large batchsize at the bottom of the script (1024 or 2048):

```
2021-04-28 20:46:22,018 - INFO - G Loss: 0.818, D Loss: 0.640, step_time: 0.046, throughput: 89142.344 img/s.
2021-04-28 20:46:22,064 - INFO - G Loss: 0.732, D Loss: 0.638, step_time: 0.046, throughput: 89052.240 img/s.
2021-04-28 20:46:22,110 - INFO - G Loss: 0.838, D Loss: 0.637, step_time: 0.046, throughput: 88636.896 img/s.
2021-04-28 20:46:22,156 - INFO - G Loss: 0.716, D Loss: 0.639, step_time: 0.046, throughput: 89192.327 img/s.
2021-04-28 20:46:22,203 - INFO - G Loss: 0.847, D Loss: 0.641, step_time: 0.046, throughput: 88712.875 img/s.
2021-04-28 20:46:22,249 - INFO - G Loss: 0.697, D Loss: 0.646, step_time: 0.046, throughput: 89043.471 img/s.
2021-04-28 20:46:22,295 - INFO - G Loss: 0.832, D Loss: 0.649, step_time: 0.046, throughput: 89171.031 img/s.
2021-04-28 20:46:22,341 - INFO - G Loss: 0.686, D Loss: 0.655, step_time: 0.046, throughput: 88623.179 img/s.
```

Additionally, some of the operations have to be re-written to stay entirely within tensorflow and not use numpy calls.

We also can enable XLA Fusion (for GPU or CPU) to speed up the computations by fusing small ops together.  

XLA can be enabled by a command line variable:
`TF_XLA_FLAGS=--tf_xla_auto_jit=2 python train_GAN_iofix.py`

This shaves a few milliseconds off the step time and gets us above 100k Img/s:
```2021-04-28 20:47:39,407 - INFO - G Loss: 0.812, D Loss: 0.669, step_time: 0.040, throughput: 101395.642 img/s.
2021-04-28 20:47:39,447 - INFO - G Loss: 0.683, D Loss: 0.667, step_time: 0.041, throughput: 101135.393 img/s.
2021-04-28 20:47:39,487 - INFO - G Loss: 0.790, D Loss: 0.664, step_time: 0.040, throughput: 102359.220 img/s.
2021-04-28 20:47:39,528 - INFO - G Loss: 0.704, D Loss: 0.661, step_time: 0.040, throughput: 102411.696 img/s.
2021-04-28 20:47:39,568 - INFO - G Loss: 0.786, D Loss: 0.656, step_time: 0.040, throughput: 102227.050 img/s.
2021-04-28 20:47:39,608 - INFO - G Loss: 0.719, D Loss: 0.653, step_time: 0.040, throughput: 101856.127 img/s.
2021-04-28 20:47:39,648 - INFO - G Loss: 0.795, D Loss: 0.649, step_time: 0.040, throughput: 102247.128 img/s.
2021-04-28 20:47:39,688 - INFO - G Loss: 0.726, D Loss: 0.646, step_time: 0.040, throughput: 102337.880 img/s.
```

Beyond this, we'll have to run the TF Profiler.  That is in the next folder.
