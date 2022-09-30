# Line by Line profiling

# Installation
 
Line profiler is, unfortunately, not part of the standard python installation.  It's also not built into the container.  Typically you can install it with:

`pip install line_profiler` but if you are using the conda module, it should be preinstalled for you.  Check with:

```bash
which kernprof
```

# Using line_profiler
The first, and most easily accessible, profiling tool, is the line profiling tool.

Run the profiling tool using `kernprof` instead of python.  This is only for single-node performance.  For example:
```bash
kernprof -l train_GAN.py
```

line_profiler produces and output file, by default named `train_GAN.py.lprof`.  You can see a line-by-line profile of your code with:


This will dump the output for 3 functions, the biggest compute users, into a file `train_GAN.py.lprof`.  Let's dump out the line by line calls:

```bash
python -m line_profiler train_GAN.py.lprof
```

First, we see that the main training function is 15.4 seconds, but 80% is the training loop.  20% is initialization overhead which is only appearing large because the total runtime is so small.

```
Total time: 17.4175 s
File: train_GAN.py
Function: train_GAN at line 417

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   417                                           @profile
   418                                           def train_GAN(_batch_size, _training_iterations, global_size):
   419
   420
   421
   422         1     951657.0 951657.0      5.5      generator = Generator()
   423
   424         1         81.0     81.0      0.0      random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
   425         1    1766091.0 1766091.0     10.1      generated_image = generator(random_input)
   426
   427
   428         1      17693.0  17693.0      0.1      discriminator = Discriminator()
   429         1      29254.0  29254.0      0.2      classification  = discriminator(generated_image)
   430
   431         1          3.0      3.0      0.0      models = {
   432         1          1.0      1.0      0.0          "generator" : generator,
   433         1          1.0      1.0      0.0          "discriminator" : discriminator
   434                                               }
   435
   436         1          1.0      1.0      0.0      opts = {
   437         1        204.0    204.0      0.0          "generator" : tf.keras.optimizers.Adam(0.001),
   438         1        142.0    142.0      0.0          "discriminator" : tf.keras.optimizers.RMSprop(0.0001)
   439
   440
   441                                               }
   442
   443         1          1.0      1.0      0.0      if global_size != 1:
   444                                                   hvd.broadcast_variables(generator.variables, root_rank=0)
   445                                                   hvd.broadcast_variables(discriminator.variables, root_rank=0)
   446                                                   hvd.broadcast_variables(opts['generator'].variables(), root_rank=0)
   447                                                   hvd.broadcast_variables(opts['discriminator'].variables(), root_rank=0)
   448
   449         1   14652412.0 14652412.0     84.1      train_loop(_batch_size, _training_iterations, models, opts, global_size)
   450
   451
   452                                               # Save the model:
   453                                               generator.save_weights("trained_GAN.h5")
```


Digging into the `train_loop` function, most of our time is spent in the `forward_pass` function so let's dig there too

```
Total time: 14.6516 s
File: train_GAN.py
Function: train_loop at line 372

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   372                                           @profile
   373                                           def train_loop(batch_size, n_training_epochs, models, opts, global_size):
   374
   375         1          9.0      9.0      0.0      logger = logging.getLogger()
   376
   377         1         25.0     25.0      0.0      rank = hvd.rank()
   378         1          1.0      1.0      0.0      for i_epoch in range(n_training_epochs):
   379
   380         1          1.0      1.0      0.0          epoch_steps = int(60000/batch_size)
   381
   382        25         31.0      1.2      0.0          for i_batch in range(epoch_steps):
   383
   384        25         30.0      1.2      0.0              start = time.time()
   385
   386        73         91.0      1.2      0.0              for network in ["generator", "discriminator"]:
   387
   388        49       1067.0     21.8      0.0                  with tf.GradientTape() as tape:
   389        98   13319553.0 135913.8     90.9                          loss = forward_pass(
   390        49         47.0      1.0      0.0                              models["generator"],
   391        49         29.0      0.6      0.0                              models["discriminator"],
   392        49         33.0      0.7      0.0                              _input_size = 100,
   393        49         34.0      0.7      0.0                              _batch_size = batch_size,
   394                                                                   )
   395
   396
   397        48         46.0      1.0      0.0                  if global_size != 1:
   398                                                               tape = hvd.DistributedGradientTape(tape)
   399
   400        48       4695.0     97.8      0.0                  if loss["discriminator"] < 0.01:
   401                                                               break
   402
   403
   404        48       9500.0    197.9      0.1                  trainable_vars = models[network].trainable_variables
   405
   406                                                           # Apply the update to the network (one at a time):
   407        48     938778.0  19557.9      6.4                  grads = tape.gradient(loss[network], trainable_vars)
   408
   409        48     314463.0   6551.3      2.1                  opts[network].apply_gradients(zip(grads, trainable_vars))
   410
   411        24         78.0      3.2      0.0              end = time.time()
   412
   413        24         22.0      0.9      0.0              images = batch_size*2*global_size
   414
   415        24      63100.0   2629.2      0.4              logger.info(f"({i_epoch}, {i_batch}), G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")
```


And in forward_pass, we see that almost all the time is fetching the real data!
```
Total time: 13.3171 s
File: train_GAN.py
Function: forward_pass at line 301

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   301                                           @profile
   302                                           def forward_pass(_generator, _discriminator, _batch_size, _input_size):
   303                                                   '''
   304                                                   This function takes the two models and runs a forward pass to the computation of the loss functions
   305                                                   '''
   306
   307                                                   # Fetch real data:
   308        49   12434313.0 253761.5     93.4          real_data = fetch_real_batch(_batch_size)
   309
   310
   311
   312                                                   # Use the generator to make fake images:
   313        48       3636.0     75.8      0.0          random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   314        48        116.0      2.4      0.0          random_noise = random_noise.reshape([_batch_size, _input_size])
   315        48     367168.0   7649.3      2.8          fake_images  = _generator(random_noise)
   316
   317
   318                                                   # Use the discriminator to make a prediction on the REAL data:
   319        48     231539.0   4823.7      1.7          prediction_on_real_data = _discriminator(real_data)
   320                                                   # Use the discriminator to make a prediction on the FAKE data:
   321        48     207829.0   4329.8      1.6          prediction_on_fake_data = _discriminator(fake_images)
   322
   323
   324        48         59.0      1.2      0.0          soften = 0.1
   325        48        707.0     14.7      0.0          real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   326        48        790.0     16.5      0.0          fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   327        48         86.0      1.8      0.0          gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   328
   329
   330                                                   # Occasionally, we disrupt the discriminator (since it has an easier job)
   331
   332                                                   # Invert a few of the discriminator labels:
   333
   334        48         84.0      1.8      0.0          n_swap = int(_batch_size * 0.1)
   335
   336        48        154.0      3.2      0.0          real_labels [0:n_swap] = 1.
   337        48         69.0      1.4      0.0          fake_labels [0:n_swap] = 0.
   338
   339
   340                                                   # Compute the loss for the discriminator on the real images:
   341        96      25143.0    261.9      0.2          discriminator_real_loss = compute_loss(
   342        48         36.0      0.8      0.0              _logits  = prediction_on_real_data,
   343        48         25.0      0.5      0.0              _targets = real_labels)
   344
   345                                                   # Compute the loss for the discriminator on the fakse images:
   346        96      20120.0    209.6      0.2          discriminator_fake_loss = compute_loss(
   347        48         39.0      0.8      0.0              _logits  = prediction_on_fake_data,
   348        48         22.0      0.5      0.0              _targets = fake_labels)
   349
   350                                                   # The generator loss is based on the output of the discriminator.
   351                                                   # It wants the discriminator to pick the fake data as real
   352        48         76.0      1.6      0.0          generator_target_labels = [1] * _batch_size
   353
   354        96      19923.0    207.5      0.1          generator_loss = compute_loss(
   355        48         28.0      0.6      0.0              _logits  = prediction_on_fake_data,
   356        48         37.0      0.8      0.0              _targets = real_labels)
   357
   358                                                   # Average the discriminator loss:
   359        48       4967.0    103.5      0.0          discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   360
   361
   362        48         42.0      0.9      0.0          loss = {
   363        48         49.0      1.0      0.0              "discriminator" : discriminator_loss,
   364        48         31.0      0.6      0.0              "generator"    : generator_loss
   365                                                   }
   366
   367        48         29.0      0.6      0.0          return loss
```


What's in that function?
```python
def fetch_real_batch(_batch_size):
    x_train, x_test = get_dataset()

    indexes = numpy.random.choice(a=x_train.shape[0], size=[_batch_size,])

    images = x_train[indexes].reshape(_batch_size, 28, 28, 1)

    return images
```

Well, we call `get_dataset` every time!  If we make the dataset a global, that we read from instead of reload, we should get a big improvement:

```2021-04-30 20:25:52,430 - INFO - (0, 23), G Loss: 0.653, D Loss: 0.658, step_time: 0.057, throughput: 2236.105 img/s.
2021-04-30 20:25:52,488 - INFO - (0, 24), G Loss: 0.666, D Loss: 0.647, step_time: 0.057, throughput: 2236.794 img/s.
2021-04-30 20:25:52,546 - INFO - (0, 25), G Loss: 0.661, D Loss: 0.653, step_time: 0.057, throughput: 2234.895 img/s.
2021-04-30 20:25:52,604 - INFO - (0, 26), G Loss: 0.667, D Loss: 0.651, step_time: 0.057, throughput: 2234.811 img/s.
2021-04-30 20:25:52,667 - INFO - (0, 27), G Loss: 0.685, D Loss: 0.652, step_time: 0.062, throughput: 2076.870 img/s.
2021-04-30 20:25:52,725 - INFO - (0, 28), G Loss: 0.710, D Loss: 0.659, step_time: 0.057, throughput: 2236.673 img/s.
2021-04-30 20:25:52,783 - INFO - (0, 29), G Loss: 0.713, D Loss: 0.655, step_time: 0.057, throughput: 2235.416 img/s.
2021-04-30 20:25:52,841 - INFO - (0, 30), G Loss: 0.732, D Loss: 0.649, step_time: 0.057, throughput: 2236.478 img/s.
2021-04-30 20:25:52,899 - INFO - (0, 31), G Loss: 0.679, D Loss: 0.650, step_time: 0.057, throughput: 2234.988 img/s.
2021-04-30 20:25:52,957 - INFO - (0, 32), G Loss: 0.746, D Loss: 0.647, step_time: 0.057, throughput: 2237.764 img/s.
2021-04-30 20:25:53,015 - INFO - (0, 33), G Loss: 0.724, D Loss: 0.647, step_time: 0.057, throughput: 2239.668 img/s.
2021-04-30 20:25:53,073 - INFO - (0, 34), G Loss: 0.717, D Loss: 0.641, step_time: 0.057, throughput: 2239.911 img/s.
2021-04-30 20:25:53,131 - INFO - (0, 35), G Loss: 0.686, D Loss: 0.637, step_time: 0.057, throughput: 2239.201 img/s.
2021-04-30 20:25:53,189 - INFO - (0, 36), G Loss: 0.767, D Loss: 0.631, step_time: 0.057, throughput: 2238.912 img/s.
2021-04-30 20:25:53,246 - INFO - (0, 37), G Loss: 0.704, D Loss: 0.636, step_time: 0.057, throughput: 2240.285 img/s.
2021-04-30 20:25:53,304 - INFO - (0, 38), G Loss: 0.703, D Loss: 0.632, step_time: 0.057, throughput: 2241.585 img/s
```

Even with the profiler still running, we're at 2000 Img/s - a 10x improvement!

We'll pick up again in the `tf_function` folder.  But first, here's a report on the line-by-line profiling:

```

Total time: 1.26872 s
File: train_GAN_iofix.py
Function: forward_pass at line 308

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   308                                           @profile
   309                                           def forward_pass(_generator, _discriminator, _real_batch, _input_size):
   310                                                   '''
   311                                                   This function takes the two models and runs a forward pass to the computation of the loss functions
   312                                                   '''
   313
   314                                                   # Fetch real data:
   315                                                   # real_data = fetch_real_batch(_batch_size)
   316        79         59.0      0.7      0.0          real_data = _real_batch
   317
   318        79        675.0      8.5      0.1          _batch_size = _real_batch.shape[0]
   319
   320                                                   # Use the generator to make fake images:
   321        79       6280.0     79.5      0.5          random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   322        79        222.0      2.8      0.0          random_noise = random_noise.reshape([_batch_size, _input_size])
   323        79     523930.0   6632.0     41.3          fake_images  = _generator(random_noise)
   324
   325
   326                                                   # Use the discriminator to make a prediction on the REAL data:
   327        79     316801.0   4010.1     25.0          prediction_on_real_data = _discriminator(real_data)
   328                                                   # Use the discriminator to make a prediction on the FAKE data:
   329        79     305338.0   3865.0     24.1          prediction_on_fake_data = _discriminator(fake_images)
   330
   331
   332        79         94.0      1.2      0.0          soften = 0.1
   333        79        974.0     12.3      0.1          real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   334        79       1150.0     14.6      0.1          fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   335        79        123.0      1.6      0.0          gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   336
   337
   338                                                   # Occasionally, we disrupt the discriminator (since it has an easier job)
   339
   340                                                   # Invert a few of the discriminator labels:
   341
   342        79        126.0      1.6      0.0          n_swap = int(_batch_size * 0.1)
   343
   344        79        229.0      2.9      0.0          real_labels [0:n_swap] = 1.
   345        79        125.0      1.6      0.0          fake_labels [0:n_swap] = 0.
   346
   347
   348                                                   # Compute the loss for the discriminator on the real images:
   349       158      38649.0    244.6      3.0          discriminator_real_loss = compute_loss(
   350        79         59.0      0.7      0.0              _logits  = prediction_on_real_data,
   351        79         48.0      0.6      0.0              _targets = real_labels)
   352
   353                                                   # Compute the loss for the discriminator on the fakse images:
   354       158      32937.0    208.5      2.6          discriminator_fake_loss = compute_loss(
   355        79         54.0      0.7      0.0              _logits  = prediction_on_fake_data,
   356        79         57.0      0.7      0.0              _targets = fake_labels)
   357
   358                                                   # The generator loss is based on the output of the discriminator.
   359                                                   # It wants the discriminator to pick the fake data as real
   360        79        134.0      1.7      0.0          generator_target_labels = [1] * _batch_size
   361
   362       158      32318.0    204.5      2.5          generator_loss = compute_loss(
   363        79         48.0      0.6      0.0              _logits  = prediction_on_fake_data,
   364        79         54.0      0.7      0.0              _targets = real_labels)
   365
   366                                                   # Average the discriminator loss:
   367        79       7988.0    101.1      0.6          discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   368
   369        79         63.0      0.8      0.0          loss = {
   370        79         81.0      1.0      0.0              "discriminator" : discriminator_loss,
   371        79         54.0      0.7      0.0              "generator"    : generator_loss
   372                                                   }
   373
   374
   375
   376        79         52.0      0.7      0.0          return loss

Total time: 3.14434 s
File: train_GAN_iofix.py
Function: train_loop at line 380

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   380                                           @profile
   381                                           def train_loop(batch_size, n_training_epochs, models, opts, global_size):
   382
   383         1         10.0     10.0      0.0      logger = logging.getLogger()
   384
   385         1         27.0     27.0      0.0      rank = hvd.rank()
   386
   387         1          2.0      2.0      0.0      for i_epoch in range(n_training_epochs):
   388
   389         1          1.0      1.0      0.0          epoch_steps = int(60000/batch_size)
   390         1        353.0    353.0      0.0          dataset.shuffle(60000) # Shuffle the whole dataset in memory
   391         1        332.0    332.0      0.0          batches = dataset.batch(batch_size=batch_size, drop_remainder=True)
   392
   393        40     150178.0   3754.4      4.8          for i_batch, batch in enumerate(batches):
   394
   395        40       9532.0    238.3      0.3              data = tf.reshape(batch, [-1, 28, 28, 1])
   396
   397        40         84.0      2.1      0.0              start = time.time()
   398
   399       118        143.0      1.2      0.0              for network in ["generator", "discriminator"]:
   400
   401        79       1521.0     19.3      0.0                  with tf.GradientTape() as tape:
   402       158    1272256.0   8052.3     40.5                          loss = forward_pass(
   403        79         80.0      1.0      0.0                              models["generator"],
   404        79         64.0      0.8      0.0                              models["discriminator"],
   405        79         55.0      0.7      0.0                              _input_size = 100,
   406        79         52.0      0.7      0.0                              _real_batch = data,
   407                                                                   )
   408
   409
   410        79         78.0      1.0      0.0                  if global_size != 1:
   411                                                               tape = hvd.DistributedGradientTape(tape)
   412
   413        79       7003.0     88.6      0.2                  if loss["discriminator"] < 0.01:
   414                                                               break
   415
   416
   417        79      15611.0    197.6      0.5                  trainable_vars = models[network].trainable_variables
   418
   419                                                           # Apply the update to the network (one at a time):
   420        79    1180893.0  14948.0     37.6                  grads = tape.gradient(loss[network], trainable_vars)
   421
   422        79     492529.0   6234.5     15.7                  opts[network].apply_gradients(zip(grads, trainable_vars))
   423
   424        39        105.0      2.7      0.0              end = time.time()
   425
   426        39         35.0      0.9      0.0              images = batch_size*2*global_size
   427
   428        39      13394.0    343.4      0.4              logger.info(f"({i_epoch}, {i_batch}), G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")

Total time: 5.34229 s
File: train_GAN_iofix.py
Function: train_GAN at line 431

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   431                                           @profile
   432                                           def train_GAN(_batch_size, _training_iterations, global_size):
   433
   434
   435         1     389211.0 389211.0      7.3      generator = Generator()
   436
   437         1         77.0     77.0      0.0      random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
   438         1    1760258.0 1760258.0     32.9      generated_image = generator(random_input)
   439
   440
   441         1      17500.0  17500.0      0.3      discriminator = Discriminator()
   442         1      29142.0  29142.0      0.5      classification  = discriminator(generated_image)
   443
   444         1          3.0      3.0      0.0      models = {
   445         1          1.0      1.0      0.0          "generator" : generator,
   446         1          1.0      1.0      0.0          "discriminator" : discriminator
   447                                               }
   448
   449         1          1.0      1.0      0.0      opts = {
   450         1        213.0    213.0      0.0          "generator" : tf.keras.optimizers.Adam(0.001),
   451         1        136.0    136.0      0.0          "discriminator" : tf.keras.optimizers.RMSprop(0.0001)
   452
   453
   454                                               }
   455
   456         1          0.0      0.0      0.0      if global_size != 1:
   457                                                   hvd.broadcast_variables(generator.variables, root_rank=0)
   458                                                   hvd.broadcast_variables(discriminator.variables, root_rank=0)
   459                                                   hvd.broadcast_variables(opts['generator'].variables(), root_rank=0)
   460                                                   hvd.broadcast_variables(opts['discriminator'].variables(), root_rank=0)
   461
   462         1    3145747.0 3145747.0     58.9      train_loop(_batch_size, _training_iterations, models, opts, global_size)
   463
   464
   465                                               # Save the model:
   466                                               generator.save_weights("trained_GAN.h5")
```

As you can see, the dominant calls are the generator, the discriminator (twice), and the gradient calculations.  We will see how to speed those up next with [`tf.function` decorators](https://github.com/argonne-lcf/CompPerfWorkshop-2021/tree/main/09_profiling_frameworks/TensorFlow/tf_function).
