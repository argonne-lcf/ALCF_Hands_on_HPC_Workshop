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

First, we see that the main training function is 20 seconds (I used CTRL+C, your time will vary but let it run for a dozen or two iterations), but more than 80% is the training loop.  Less than 20% is initialization overhead which is only appearing large because the total runtime is so small.

```
Total time: 21.6437 s
File: train_GAN.py
Function: train_GAN at line 379

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   379                                           @profile
   380                                           def train_GAN(_batch_size, _training_iterations):
   381                                           
   382                                           
   383                                           
   384         1    1685106.0 1685106.0      7.8      generator = Generator()
   385                                           
   386         1         72.0     72.0      0.0      random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
   387         1    1447876.0 1447876.0      6.7      generated_image = generator(random_input)
   388                                           
   389                                           
   390         1      96993.0  96993.0      0.4      discriminator = Discriminator()
   391         1      74967.0  74967.0      0.3      classification  = discriminator(generated_image)
   392                                           
   393         1          1.0      1.0      0.0      models = {
   394         1          1.0      1.0      0.0          "generator" : generator,
   395         1          0.0      0.0      0.0          "discriminator" : discriminator
   396                                               }
   397                                           
   398         1          1.0      1.0      0.0      opts = {
   399         1        132.0    132.0      0.0          "generator" : tf.keras.optimizers.Adam(0.001),
   400         1        107.0    107.0      0.0          "discriminator" : tf.keras.optimizers.RMSprop(0.0001)
   401                                           
   402                                           
   403                                               }
   404                                           
   405         1   18338474.0 18338474.0     84.7      train_loop(_batch_size, _training_iterations, models, opts)

```


Digging into the `train_loop` function, most of our time is spent in the `forward_pass` function so let's dig there too

```
Total time: 18.3375 s
File: train_GAN.py
Function: train_loop at line 338

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   338                                           @profile
   339                                           def train_loop(batch_size, n_training_epochs, models, opts):
   340                                           
   341         1          5.0      5.0      0.0      logger = logging.getLogger()
   342                                           
   343         1          1.0      1.0      0.0      for i_epoch in range(n_training_epochs):
   344                                           
   345         1          1.0      1.0      0.0          epoch_steps = int(60000/batch_size)
   346                                           
   347        38         43.0      1.1      0.0          for i_batch in range(40):  # range(epoch_steps):
   348                                           
   349        38         38.0      1.0      0.0              start = time.time()
   350                                           
   351       113        136.0      1.2      0.0              for network in ["generator", "discriminator"]:
   352                                           
   353        76        881.0     11.6      0.0                  with tf.GradientTape() as tape:
   354       152   16296935.0 107216.7     88.9                          loss = forward_pass(
   355        76         62.0      0.8      0.0                              models["generator"],
   356        76         46.0      0.6      0.0                              models["discriminator"],
   357        76         40.0      0.5      0.0                              _input_size = 100,
   358        76         34.0      0.4      0.0                              _batch_size = batch_size,
   359                                                                   )
   360                                           
   361                                           
   362        76      11481.0    151.1      0.1                  if loss["discriminator"] < 0.01:
   363                                                               break
   364                                           
   365                                           
   366        76      13774.0    181.2      0.1                  trainable_vars = models[network].trainable_variables
   367                                           
   368                                                           # Apply the update to the network (one at a time):
   369        76    1296831.0  17063.6      7.1                  grads = tape.gradient(loss[network], trainable_vars)
   370                                           
   371        75     707991.0   9439.9      3.9                  opts[network].apply_gradients(zip(grads, trainable_vars))
   372                                           
   373        37         83.0      2.2      0.0              end = time.time()
   374                                           
   375        37         31.0      0.8      0.0              images = batch_size*2
   376                                           
   377        37       9121.0    246.5      0.0              logger.info(f"({i_epoch}, {i_batch}), G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")

```


And in forward_pass, we see that almost all the time is fetching the real data!
```
Total time: 16.293 s
File: train_GAN.py
Function: forward_pass at line 267

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   267                                           @profile
   268                                           def forward_pass(_generator, _discriminator, _batch_size, _input_size):
   269                                               '''
   270                                               This function takes the two models and runs a forward pass to the computation of the loss functions
   271                                               '''
   272                                           
   273                                               # Fetch real data:
   274        76   14499566.0 190783.8     89.0      real_data = fetch_real_batch(_batch_size)
   275                                           
   276                                           
   277                                           
   278                                               # Use the generator to make fake images:
   279        76       5014.0     66.0      0.0      random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   280        76        237.0      3.1      0.0      random_noise = random_noise.reshape([_batch_size, _input_size])
   281        76     692202.0   9107.9      4.2      fake_images  = _generator(random_noise)
   282                                           
   283                                           
   284                                               # Use the discriminator to make a prediction on the REAL data:
   285        76     467460.0   6150.8      2.9      prediction_on_real_data = _discriminator(real_data)
   286                                               # Use the discriminator to make a prediction on the FAKE data:
   287        76     389093.0   5119.6      2.4      prediction_on_fake_data = _discriminator(fake_images)
   288                                           
   289                                           
   290        76         96.0      1.3      0.0      soften = 0.1
   291        76       1019.0     13.4      0.0      real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   292        76       1318.0     17.3      0.0      fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   293        76        107.0      1.4      0.0      gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   294                                           
   295                                           
   296                                               # Occasionally, we disrupt the discriminator (since it has an easier job)
   297                                           
   298                                               # Invert a few of the discriminator labels:
   299                                           
   300        76        113.0      1.5      0.0      n_swap = int(_batch_size * 0.1)
   301                                           
   302        76        208.0      2.7      0.0      real_labels [0:n_swap] = 1.
   303        76         90.0      1.2      0.0      fake_labels [0:n_swap] = 0.
   304                                           
   305                                           
   306                                               # Compute the loss for the discriminator on the real images:
   307       152      93340.0    614.1      0.6      discriminator_real_loss = compute_loss(
   308        76         43.0      0.6      0.0          _logits  = prediction_on_real_data,
   309        76         40.0      0.5      0.0          _targets = real_labels)
   310                                           
   311                                               # Compute the loss for the discriminator on the fakse images:
   312       152      64365.0    423.5      0.4      discriminator_fake_loss = compute_loss(
   313        76         59.0      0.8      0.0          _logits  = prediction_on_fake_data,
   314        76         39.0      0.5      0.0          _targets = fake_labels)
   315                                           
   316                                               # The generator loss is based on the output of the discriminator.
   317                                               # It wants the discriminator to pick the fake data as real
   318        76        147.0      1.9      0.0      generator_target_labels = [1] * _batch_size
   319                                           
   320       152      62160.0    408.9      0.4      generator_loss = compute_loss(
   321        76         41.0      0.5      0.0          _logits  = prediction_on_fake_data,
   322        76         38.0      0.5      0.0          _targets = real_labels)
   323                                           
   324                                               # Average the discriminator loss:
   325        76      15940.0    209.7      0.1      discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   326                                           
   327                                           
   328        76         58.0      0.8      0.0      loss = {
   329        76         91.0      1.2      0.0          "discriminator" : discriminator_loss,
   330        76         43.0      0.6      0.0          "generator"    : generator_loss
   331                                               }
   332                                           
   333        76         42.0      0.6      0.0      return loss

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

```INFO:root:(0, 25), G Loss: 0.699, D Loss: 0.651, step_time: 0.082, throughput: 1552.844 img/s.
INFO:root:(0, 26), G Loss: 0.654, D Loss: 0.657, step_time: 0.082, throughput: 1553.136 img/s.
INFO:root:(0, 27), G Loss: 0.688, D Loss: 0.666, step_time: 0.082, throughput: 1552.588 img/s.
INFO:root:(0, 28), G Loss: 0.672, D Loss: 0.661, step_time: 0.082, throughput: 1569.858 img/s.
INFO:root:(0, 29), G Loss: 0.659, D Loss: 0.665, step_time: 0.081, throughput: 1577.682 img/s.
INFO:root:(0, 30), G Loss: 0.651, D Loss: 0.663, step_time: 0.081, throughput: 1576.135 img/s.
INFO:root:(0, 31), G Loss: 0.673, D Loss: 0.656, step_time: 0.081, throughput: 1571.572 img/s.
INFO:root:(0, 32), G Loss: 0.685, D Loss: 0.655, step_time: 0.081, throughput: 1574.129 img/s.
INFO:root:(0, 33), G Loss: 0.682, D Loss: 0.654, step_time: 0.082, throughput: 1565.564 img/s.
INFO:root:(0, 34), G Loss: 0.732, D Loss: 0.648, step_time: 0.082, throughput: 1569.908 img/s.
INFO:root:(0, 35), G Loss: 0.676, D Loss: 0.648, step_time: 0.082, throughput: 1569.541 img/s.
INFO:root:(0, 36), G Loss: 0.685, D Loss: 0.646, step_time: 0.082, throughput: 1568.367 img/s.
INFO:root:(0, 37), G Loss: 0.685, D Loss: 0.640, step_time: 0.081, throughput: 1577.255 img/s.
INFO:root:(0, 38), G Loss: 0.677, D Loss: 0.643, step_time: 0.081, throughput: 1579.957 img/s.
INFO:root:(0, 39), G Loss: 0.706, D Loss: 0.637, step_time: 0.081, throughput: 1578.559 img/s.

```

Even with the profiler still running, we're at 1500 Img/s - a 5x improvement!

We'll pick up again in the `tf_function` folder.  But first, here's a report on the line-by-line profiling:

```
$ python -m line_profiler train_GAN_iofix.py.lprof
Timer unit: 1e-06 s

Total time: 1.85503 s
File: train_GAN_iofix.py
Function: forward_pass at line 277

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   277                                           @profile
   278                                           def forward_pass(_generator, _discriminator, _batch_size, _input_size):
   279                                               '''
   280                                               This function takes the two models and runs a forward pass to the computation of the loss functions
   281                                               '''
   282                                           
   283                                               # Fetch real data:
   284        80     126706.0   1583.8      6.8      real_data = fetch_real_batch(_batch_size)
   285                                           
   286                                           
   287                                           
   288                                               # Use the generator to make fake images:
   289        80       5209.0     65.1      0.3      random_noise = numpy.random.uniform(-1, 1, size=_batch_size*_input_size).astype(numpy.float32)
   290        80        173.0      2.2      0.0      random_noise = random_noise.reshape([_batch_size, _input_size])
   291        80     623508.0   7793.9     33.6      fake_images  = _generator(random_noise)
   292                                           
   293                                           
   294                                               # Use the discriminator to make a prediction on the REAL data:
   295        80     452449.0   5655.6     24.4      prediction_on_real_data = _discriminator(real_data)
   296                                               # Use the discriminator to make a prediction on the FAKE data:
   297        80     403735.0   5046.7     21.8      prediction_on_fake_data = _discriminator(fake_images)
   298                                           
   299                                           
   300        80        100.0      1.2      0.0      soften = 0.1
   301        80        780.0      9.8      0.0      real_labels = numpy.zeros([_batch_size,1], dtype=numpy.float32) + soften
   302        80        965.0     12.1      0.1      fake_labels = numpy.ones([_batch_size,1],  dtype=numpy.float32) - soften
   303        80        105.0      1.3      0.0      gen_labels  = numpy.zeros([_batch_size,1], dtype=numpy.float32)
   304                                           
   305                                           
   306                                               # Occasionally, we disrupt the discriminator (since it has an easier job)
   307                                           
   308                                               # Invert a few of the discriminator labels:
   309                                           
   310        80         97.0      1.2      0.0      n_swap = int(_batch_size * 0.1)
   311                                           
   312        80        194.0      2.4      0.0      real_labels [0:n_swap] = 1.
   313        80         88.0      1.1      0.0      fake_labels [0:n_swap] = 0.
   314                                           
   315                                           
   316                                               # Compute the loss for the discriminator on the real images:
   317       160      89045.0    556.5      4.8      discriminator_real_loss = compute_loss(
   318        80         42.0      0.5      0.0          _logits  = prediction_on_real_data,
   319        80         42.0      0.5      0.0          _targets = real_labels)
   320                                           
   321                                               # Compute the loss for the discriminator on the fakse images:
   322       160      68763.0    429.8      3.7      discriminator_fake_loss = compute_loss(
   323        80         54.0      0.7      0.0          _logits  = prediction_on_fake_data,
   324        80         46.0      0.6      0.0          _targets = fake_labels)
   325                                           
   326                                               # The generator loss is based on the output of the discriminator.
   327                                               # It wants the discriminator to pick the fake data as real
   328        80        158.0      2.0      0.0      generator_target_labels = [1] * _batch_size
   329                                           
   330       160      66151.0    413.4      3.6      generator_loss = compute_loss(
   331        80         43.0      0.5      0.0          _logits  = prediction_on_fake_data,
   332        80         51.0      0.6      0.0          _targets = real_labels)
   333                                           
   334                                               # Average the discriminator loss:
   335        80      16275.0    203.4      0.9      discriminator_loss = 0.5*(discriminator_fake_loss  + discriminator_real_loss)
   336                                           
   337                                           
   338        80         62.0      0.8      0.0      loss = {
   339        80         94.0      1.2      0.0          "discriminator" : discriminator_loss,
   340        80         47.0      0.6      0.0          "generator"    : generator_loss
   341                                               }
   342                                           
   343        80         44.0      0.6      0.0      return loss

Total time: 3.98949 s
File: train_GAN_iofix.py
Function: train_loop at line 347

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   347                                           @profile
   348                                           def train_loop(batch_size, n_training_epochs, models, opts):
   349                                           
   350         1          4.0      4.0      0.0      logger = logging.getLogger()
   351                                           
   352         2          2.0      1.0      0.0      for i_epoch in range(n_training_epochs):
   353                                           
   354         1          1.0      1.0      0.0          epoch_steps = int(60000/batch_size)
   355                                           
   356        41         42.0      1.0      0.0          for i_batch in range(40):  # range(epoch_steps):
   357                                           
   358        40         35.0      0.9      0.0              start = time.time()
   359                                           
   360       120        132.0      1.1      0.0              for network in ["generator", "discriminator"]:
   361                                           
   362        80        701.0      8.8      0.0                  with tf.GradientTape() as tape:
   363       160    1858327.0  11614.5     46.6                          loss = forward_pass(
   364        80         48.0      0.6      0.0                              models["generator"],
   365        80         38.0      0.5      0.0                              models["discriminator"],
   366        80         40.0      0.5      0.0                              _input_size = 100,
   367        80         43.0      0.5      0.0                              _batch_size = batch_size,
   368                                                                   )
   369                                           
   370                                           
   371        80      10828.0    135.3      0.3                  if loss["discriminator"] < 0.01:
   372                                                               break
   373                                           
   374                                           
   375        80      13392.0    167.4      0.3                  trainable_vars = models[network].trainable_variables
   376                                           
   377                                                           # Apply the update to the network (one at a time):
   378        80    1345956.0  16824.5     33.7                  grads = tape.gradient(loss[network], trainable_vars)
   379                                           
   380        80     751751.0   9396.9     18.8                  opts[network].apply_gradients(zip(grads, trainable_vars))
   381                                           
   382        40         47.0      1.2      0.0              end = time.time()
   383                                           
   384        40         32.0      0.8      0.0              images = batch_size*2
   385                                           
   386        40       8075.0    201.9      0.2              logger.info(f"({i_epoch}, {i_batch}), G Loss: {loss['generator']:.3f}, D Loss: {loss['discriminator']:.3f}, step_time: {end-start :.3f}, throughput: {images/(end-start):.3f} img/s.")

Total time: 5.60629 s
File: train_GAN_iofix.py
Function: train_GAN at line 388

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
   388                                           @profile
   389                                           def train_GAN(_batch_size, _training_iterations):
   390                                           
   391                                           
   392                                           
   393         1      25489.0  25489.0      0.5      generator = Generator()
   394                                           
   395         1         61.0     61.0      0.0      random_input = numpy.random.uniform(-1,1,[1,100]).astype(numpy.float32)
   396         1    1418910.0 1418910.0     25.3      generated_image = generator(random_input)
   397                                           
   398                                           
   399         1      92874.0  92874.0      1.7      discriminator = Discriminator()
   400         1      78218.0  78218.0      1.4      classification  = discriminator(generated_image)
   401                                           
   402         1          0.0      0.0      0.0      models = {
   403         1          1.0      1.0      0.0          "generator" : generator,
   404         1          0.0      0.0      0.0          "discriminator" : discriminator
   405                                               }
   406                                           
   407         1          1.0      1.0      0.0      opts = {
   408         1        135.0    135.0      0.0          "generator" : tf.keras.optimizers.Adam(0.001),
   409         1        109.0    109.0      0.0          "discriminator" : tf.keras.optimizers.RMSprop(0.0001)
   410                                           
   411                                           
   412                                               }
   413                                           
   414         1    3990489.0 3990489.0     71.2      train_loop(_batch_size, _training_iterations, models, opts)


```

As you can see, the dominant calls are the generator, the discriminator (twice), and the gradient calculations.  We will see how to speed those up next with [`tf.function` decorators](https://github.com/argonne-lcf/CompPerfWorkshop-2021/tree/main/09_profiling_frameworks/TensorFlow/tf_function).
