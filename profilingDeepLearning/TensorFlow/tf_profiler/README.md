# TF Profiling Tool


Note the lines `tf.profiler.experimental.start('logdir')` and `tf.profiler.experimental.stop()` in the code.  This sets up and tears down the profiling tool built in to tensorflow.  See the screenshots below - the main operation is conv2D backprop - a very compute heavy operation.  We may get some performance improvement further with reduced precision - see the `reduced_precision` folder.


# Running the tf profiler

When you've captured your profile data, tensorboard will dump it into the folder `logdir` (as above) and you will have to view it.  The simplest way, for this application, is to copy it to your own laptop if you have tensorflow installed.  If not, you can run tensorboard on Theta and use an ssh port forward to view it on your own laptop.

Whatever you do, you can open tensorboard like so:
`tensorboard --logdir [your/own/path/to/logdir/]`

Next, open your browser and navigate to `localhost:6006` (or, whatever port you forwarded to) and you'll see a screen like the one below:

![Tensorboard Profiler Overview](profiler_overview.png)

#### *(Port forwarding for tensorboard)*
*(If you want to run tensorboard directly on compute node at ALCF you will need to do port forwarding*
```
ssh -L PORT:localhost:PORT USER@theta.alcf.anl.gov ssh -L PORT:localhost:PORT thetagpusn1 ssh -L PORT:localhost:PORT thetagpuXXX
```
*where `thetagpuXXX` is the name of compute node where tensorboard is running.)*

And, if you scroll down, you'll see the list of the top 10 most time consuming operations:

![top 10](top10_ops.png)

This list shows us that the top operations are largely all convolution ops (particularly backprop).  The profiler at the top also points out that 0% of the graph is in reduced precision, which could give us a speedup.  We'll try that next but first let's review the other tabs:

Here is the Kernel Statistics page:

![kernel stats](kernel-stats.png)

Again, this shows that the convolution operations are all the most dominant and equally distributed (roughly).

The tensorflow statistics shows similar info:

![tf stats](tf-stats.png)

And there is also a timeline view of all ops (trace viewer)

![timeline](trace-viewer.png)

And zoomed:

![timeline zoom](trace-viewer-zoom.png)

Now, let's try running in reduced precision.
