# Reduced Precision

Switching to reduced precision is not that hard in TensorFlow:

```python
tf.keras.mixed_precision.set_global_policy("mixed_float16")
```

In this case, there are also a bunch of places where I hard-coded `float32` - oops!  Fix those too.

Next, run with mixed precision (profiling is still turned on here):

```
INFO:root:(9, 16), G Loss: 0.754, D Loss: 0.615, step_time: 0.025, throughput: 162552.695 img/s.
INFO:root:(9, 17), G Loss: 0.762, D Loss: 0.620, step_time: 0.025, throughput: 162734.387 img/s.
INFO:root:(9, 18), G Loss: 0.755, D Loss: 0.629, step_time: 0.025, throughput: 163159.402 img/s.
INFO:root:(9, 19), G Loss: 0.777, D Loss: 0.639, step_time: 0.025, throughput: 162735.928 img/s.
INFO:root:(9, 20), G Loss: 0.752, D Loss: 0.637, step_time: 0.025, throughput: 162651.189 img/s.
INFO:root:(9, 21), G Loss: 0.848, D Loss: 0.644, step_time: 0.025, throughput: 162032.965 img/s.
INFO:root:(9, 22), G Loss: 0.683, D Loss: 0.649, step_time: 0.025, throughput: 162702.022 img/s.
INFO:root:(9, 23), G Loss: 1.082, D Loss: 0.664, step_time: 0.025, throughput: 163047.910 img/s.
INFO:root:(9, 24), G Loss: 0.592, D Loss: 0.671, step_time: 0.025, throughput: 162345.324 img/s.
INFO:root:(9, 25), G Loss: 1.044, D Loss: 0.648, step_time: 0.025, throughput: 162668.130 img/s.
INFO:root:(9, 26), G Loss: 0.707, D Loss: 0.638, step_time: 0.025, throughput: 161794.912 img/s.
INFO:root:(9, 27), G Loss: 0.949, D Loss: 0.625, step_time: 0.025, throughput: 162225.750 img/s.
INFO:root:(9, 28), G Loss: 0.778, D Loss: 0.621, step_time: 0.025, throughput: 163931.614 img/s.
```

This is disappointing!  We ran with mixed precision and it is NOT FASTER.  Let's look into the profile to discover why.

Here's the overview page.  We note right away that in the bottom left, it IS using a good amount of reduced precision.

![Tensorboard Profiler Overview](profiler_overview.png)

Scrolling down:

![top 10](top10-ops.png)

Compared to the float32 top-10 operations, this is pretty different.  There is one op that is particularly dominant!  It is still a conv2d operation, but for some reason it is much slower than the others.

Here is the Kernel Statistics page again:

![kernel stats](kernel-stats.png)

We see the same problem there - except this time it's pointing to the wgrad convolution. The tensorflow statistics shows similar info:

![tf stats](tf-stats.png)

And there is also a timeline view of all ops (trace viewer)

![timeline](trace-viewer.png)

And zoomed:

![timeline zoom](trace-zoomed.png)

So, reduced precision appears to be slower because of one particular operation.  How to fix this?  Well the best solution here is probably to open a bug report.
