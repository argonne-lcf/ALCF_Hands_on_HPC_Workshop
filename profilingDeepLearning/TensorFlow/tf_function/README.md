# TF Function and Graph Compilation

Using line profiler showed us that the largest computation use, by far, was the train loop and is subcalls.  Here, we'll wrap those functions in `@tf.function` decorators to improve performance with graph compilation.  This has a mild overhead at the start for tracing but then runs more quickly.  Note the first few iterations are slower.

Additionally, some of the operations have to be re-written to stay entirely within tensorflow and not use numpy calls.


Out of the box, we see a dramatic speed up:

```INFO:root:(0, 922), G Loss: 0.791, D Loss: 0.654, step_time: 0.004, throughput: 30932.871 img/s.
INFO:root:(0, 923), G Loss: 0.634, D Loss: 0.668, step_time: 0.004, throughput: 31701.855 img/s.
INFO:root:(0, 924), G Loss: 0.798, D Loss: 0.643, step_time: 0.004, throughput: 31840.989 img/s.
INFO:root:(0, 925), G Loss: 0.707, D Loss: 0.629, step_time: 0.004, throughput: 31569.500 img/s.
INFO:root:(0, 926), G Loss: 0.758, D Loss: 0.653, step_time: 0.004, throughput: 31882.589 img/s.
INFO:root:(0, 927), G Loss: 0.708, D Loss: 0.628, step_time: 0.004, throughput: 31614.116 img/s.
INFO:root:(0, 928), G Loss: 0.789, D Loss: 0.616, step_time: 0.004, throughput: 31933.792 img/s.
INFO:root:(0, 929), G Loss: 0.687, D Loss: 0.631, step_time: 0.004, throughput: 31184.416 img/s.
INFO:root:(0, 930), G Loss: 0.800, D Loss: 0.630, step_time: 0.004, throughput: 31478.799 img/s.
INFO:root:(0, 931), G Loss: 0.716, D Loss: 0.622, step_time: 0.004, throughput: 31914.809 img/s.
INFO:root:(0, 932), G Loss: 0.846, D Loss: 0.620, step_time: 0.004, throughput: 32169.148 img/s.
INFO:root:(0, 933), G Loss: 0.616, D Loss: 0.650, step_time: 0.004, throughput: 31837.212 img/s.
INFO:root:(0, 934), G Loss: 0.993, D Loss: 0.638, step_time: 0.004, throughput: 31425.364 img/s.
INFO:root:(0, 935), G Loss: 0.580, D Loss: 0.662, step_time: 0.004, throughput: 31775.030 img/s.

```

Instead of 230 Img/s, we're at 30000+ Img/s.  But note that the step time is now only 0.004 seconds.  We're likely not doing enough work per batch.  Change it to a large batchsize at the bottom of the script (1024 or 2048):

```
INFO:root:(0, 16), G Loss: 0.597, D Loss: 0.646, step_time: 0.047, throughput: 86891.209 img/s.
INFO:root:(0, 17), G Loss: 0.606, D Loss: 0.655, step_time: 0.047, throughput: 86948.378 img/s.
INFO:root:(0, 18), G Loss: 0.624, D Loss: 0.658, step_time: 0.047, throughput: 86724.092 img/s.
INFO:root:(0, 19), G Loss: 0.641, D Loss: 0.659, step_time: 0.047, throughput: 86969.506 img/s.
INFO:root:(0, 20), G Loss: 0.657, D Loss: 0.658, step_time: 0.047, throughput: 87070.002 img/s.
INFO:root:(0, 21), G Loss: 0.662, D Loss: 0.658, step_time: 0.047, throughput: 87027.219 img/s.
INFO:root:(0, 22), G Loss: 0.674, D Loss: 0.660, step_time: 0.047, throughput: 87076.622 img/s.
INFO:root:(0, 23), G Loss: 0.663, D Loss: 0.662, step_time: 0.047, throughput: 86923.302 img/s.
INFO:root:(0, 24), G Loss: 0.664, D Loss: 0.662, step_time: 0.047, throughput: 87160.117 img/s.
INFO:root:(0, 25), G Loss: 0.664, D Loss: 0.661, step_time: 0.047, throughput: 86706.584 img/s.
INFO:root:(0, 26), G Loss: 0.667, D Loss: 0.660, step_time: 0.047, throughput: 86907.473 img/s.
INFO:root:(0, 27), G Loss: 0.652, D Loss: 0.655, step_time: 0.047, throughput: 87014.436 img/s.
INFO:root:(0, 28), G Loss: 0.664, D Loss: 0.650, step_time: 0.047, throughput: 86969.065 img/s.

```


We also can enable XLA Fusion (for GPU or CPU) to speed up the computations by fusing small ops together.  

XLA can be enabled by a command line variable:
`TF_XLA_FLAGS=--tf_xla_auto_jit=2 python train_GAN_iofix.py`

This shaves a few milliseconds off the step time and gets us above 100k Img/s:
```
INFO:root:(0, 15), G Loss: 0.674, D Loss: 0.614, step_time: 0.038, throughput: 108926.383 img/s.
INFO:root:(0, 16), G Loss: 0.652, D Loss: 0.620, step_time: 0.038, throughput: 108900.836 img/s.
INFO:root:(0, 17), G Loss: 0.654, D Loss: 0.621, step_time: 0.038, throughput: 108983.736 img/s.
INFO:root:(0, 18), G Loss: 0.632, D Loss: 0.621, step_time: 0.038, throughput: 108932.599 img/s.
INFO:root:(0, 19), G Loss: 0.645, D Loss: 0.621, step_time: 0.038, throughput: 108921.549 img/s.
INFO:root:(0, 20), G Loss: 0.639, D Loss: 0.618, step_time: 0.038, throughput: 108364.362 img/s.
INFO:root:(0, 21), G Loss: 0.640, D Loss: 0.619, step_time: 0.038, throughput: 108419.755 img/s.
INFO:root:(0, 22), G Loss: 0.658, D Loss: 0.616, step_time: 0.038, throughput: 108621.977 img/s.
INFO:root:(0, 23), G Loss: 0.662, D Loss: 0.614, step_time: 0.038, throughput: 108740.920 img/s.
INFO:root:(0, 24), G Loss: 0.674, D Loss: 0.610, step_time: 0.038, throughput: 108900.836 img/s.
INFO:root:(0, 25), G Loss: 0.684, D Loss: 0.607, step_time: 0.038, throughput: 108814.616 img/s.
INFO:root:(0, 26), G Loss: 0.697, D Loss: 0.602, step_time: 0.038, throughput: 108881.511 img/s.
INFO:root:(0, 27), G Loss: 0.702, D Loss: 0.601, step_time: 0.038, throughput: 108840.812 img/s.
INFO:root:(0, 28), G Loss: 0.705, D Loss: 0.595, step_time: 0.038, throughput: 108805.657 img/s.
```

Beyond this, we'll have to run the TF Profiler.  That is in the next folder.
