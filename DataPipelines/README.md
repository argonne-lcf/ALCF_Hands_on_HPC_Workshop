# Building a CPU-side data pipeline

Author: J. Taylor Childers (jchilders@anl.gov)

## Learning Goals:
- Using the CPU on a system to build data batches in parallel as ML calculations are performed on the GPU
- Using parallel processes on CPU to speed up the data pipeline process.
- Do all this using the framework's (PyTorch, Tensorflow) data APIs

New AI systems largely depend on CPU-GPU hybrid architectures. This makes efficient use of CPU-side resources important in order to feed sufficient data to the GPU algorithms. Ideally, the CPU processes data and builds training batches, while the GPU performs the compute intensive forward and backward gradient calculations.

Here there are examples of building a data pipeline for both Tensorflow and PyTorch. Tensorflow's data pipeline API is a bit more advanced than PyTorch so we'll focus on that one, though we include an example in PyTorch.

# ImageNet Dataset

This example uses the ImageNet dataset to build training batches.

![Turtle](images/n01667778_12001.JPEG) ![Dog](images/n02094114_1205.JPEG)

This dataset includes JPEG images and an XML annotation for each file that defines a bounding box for each class. Building a training batch requires pre-processing the images and annotations. In our example, we have created text files that list all the files in the training set and validation set. For each text file, we need to use the input JPEG files and build tensors that include multiple images per training batch.

# Tensorflow Dataset example

Tensorflow has some very nice tools to help us build the pipeline. You'll find the [example here](00_tensorflowDatasetAPI/ilsvrc_dataset.py).

## Build from file list
We'll start in the function `build_dataset_from_filelist`.

1. Open the filelist
```python
# loading full filelist
filelist = []
with open(filelist_filename) as file:
   for line in file:
      filelist.append(line.strip())
```
2. Parse the list of files into a TF Tensor
```python
filelist = tf.data.Dataset.from_tensor_slices(filelist)
```
3. If we are using Horovod for MPI parallelism, we want to "shard" the data across nodes so each node processes unique data
```python
filelist = filelist.shard(config['hvd'].size(), config['hvd'].rank())
```
4. Shuffle our filelist at each epoch barrier
```python
filelist = filelist.shuffle(dc['shuffle_buffer'],reshuffle_each_iteration=dc['reshuffle_each_iteration'])
```
5. Run a custom function on the filelist, which effectively opens the JPEG file, loads the data into a TF Tensor and extracts the class labels. If there are multiple objects in the image, this function will return more than one image using the bounding boxes. `num_parallel_calls` allows this function to run in parallel so many JPEG files can be read into memory and processed in parallel threads.
```python
ds = filelist.map(load_image_label_bb,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
```
6. Since the previous map fuction may return one or more images, we need to unbatch the output before we batch it into our fixed batch size
```python
ds = ds.apply(tf.data.Dataset.unbatch)
ds = ds.batch(dc['batch_size'])
```
7. Tell the dataset it can prepare the next batch(es) prior to them being requested
```python
ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
```

Done.

We can now iterate over this dataset in a loop:
```python
for inputs,labels in ds:
   prediction = model(inputs)
   loss = loss_func(prediction,labels)
   # ...
```

## Parallel Processing on Polaris

The example `00_tensorflowDatasetAPI/ilsvrc_dataset.py` can be run via
```bash
cd 00_tensorflowDatasetAPI
qsub -A <project> -q debug submit_polaris.sh
```   

This script will run the example 3 times with 1 thread (no parallelism), 16 threads, and 64 threads per MPI process. The reported `imgs/sec` throughput will be lowest for serial processing and highest for the 64 threads per MPI process. You can see in this screenshot from the [Tensorflow Profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras) how processes are being utilized. 

This profile shows the single process handling all the data pipeline processes. All data pipeline calls are being done serially when they could be done in parallel. It takes over 3 seconds to prepare a batch of images.
![serial](images/ilsvrc_serial.png)

In the case of 64-threads per MPI process, batch processing time is down to 0.08 seconds. The profiler shows we are running with our 64 parallel processes, all of which are opening JPEGs, processing them into tensors, extracting truth information, and so on. Once can see the `ReadFile` operation taking place in parallel which opens the jpeg and reads in the data to memory. This operation is the most time consuming in this pipeline and by parallelizing it, we have improved our throughput.
![parallel](images/ilsvrc_64threads_zoom.png)


