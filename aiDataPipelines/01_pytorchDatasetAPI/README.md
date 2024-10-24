# PyTorch Example Data Pipeline
by J. Taylor Childers (jchilders@anl.gov)

This tutorial demonstrates how to build a parallel data pipeline in PyTorch using the Python `threading` module. The dataset used is the ImageNet dataset described in the [README](../README.md) one level up. The script uses [PyTorch Distributed Data Parallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) to distribute the training task across multiple nodes. We do not cover how DDP works in this tutorial.

Submit to Polaris using:
```bash
qsub -A <project> -q <queue> submit_polaris.sh
```

All log files and profiles go into the `logdir` folder.


## Serial Data Pipeline and Training

In [imagenet_serial.py](./imagenet_serial.py) the data pipeline is written to be serial such that the data manipulation and batch formation are all done before each training loop execution. The point is to provide an initial measure of serial performance.

When run using the `submit_polaris.sh` the output looks like this:
```shell
Batch size: 64
step_img_rate: 32.91
step_img_rate: 41.77
step_img_rate: 36.01
step_img_rate: 37.57
step_img_rate: 27.94
step_img_rate: 38.88
step_img_rate: 36.97
step_img_rate: 36.96
step_img_rate: 38.08
step_img_rate: 11.67
Average image rate: mean: 33.01, stddev: 8.68
All Done; total runtime: 49.58
```

We'll use this to compare to the parallel data pipeline.

## Parallel Data Pipeline

In the [`imagenet_parallel.py`](./imagenet_parallel.py) script you will find an implementation of a parallel data pipeline which uses the Python[`threading`](https://docs.python.org/3/library/threading.html) module. The diagram below depicts the parts of this parallel pipeline. 

### Parallel Data Pipeline Diagram
![paralle data pipeline](../images/pytorch_threading.png)

Multiple `_worker_thread` threads are assigned a sub-range of the full filelist of images. They loop over their assigned image files, load them, convert them to pytorch tensors, extract their image class ID, and place these data on a _queue_ which acts as a FIFO. There can be many of the these threads opening files in parallel and placing them on the queue.

The single `_batch_collector_thread` thread retrieves images/class-id pairs from the `image_queue` and contructs batches for training. Completed batches are placed on the `batch_queue` which are then retrieved during the training loop via the `__iter__()` call.

The handy thing about queues is that you can specify how long they can be. That way we can configure how many images can be held on the image queue, or batches on the batch queue. Once the queues are full, our threads will block on future attempts to place items in these queues until there are items removed.

## Code Breakdown

The `ImageNetDataset` class object handles our image files, loading the JPEGs, extracting the class ID from the file name, and transforming them into uniform PyTorch Tensors.

```python
# dataset handler for input files
class ImageNetDataset:
    def __init__(self, base_dir, file_list_path, id_to_index, transform=None, rank=0, world_size=1):
        self.base_dir = base_dir
        self.transform = transform
        self.id_to_index = id_to_index
        with open(file_list_path, 'r') as file:
            self.image_paths = [line.strip() for line in file]
        
        # Partition the dataset among ranks
        self.image_paths = self.image_paths[rank::world_size]

    def __len__(self):
        return len(self.image_paths)

    def load_image(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        unique_id = img_path.split('/')[-2]
        target = self.id_to_index[unique_id]
        if self.transform:
            img = self.transform(img)
        return img, target
```

The `ImageNetDataLoader` creates the queues for images and batches, and launches our worker and batch threads. It also knows how to stop the threads on command.

```python
# data loader class which spawns multiple threads to load images and a single thread to batch them.
class ImageNetDataLoader:
    def __init__(self, dataset, batch_size, num_workers, cache_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size

        self.image_queue = queue.Queue(maxsize=2 * batch_size * num_workers)
        self.batch_queue = queue.Queue(maxsize=cache_size)

        self.threads = []
        self.stop_event = threading.Event()

        for _ in range(num_workers):
            thread = threading.Thread(target=self._worker_thread)
            thread.start()
            self.threads.append(thread)

        self.batch_thread = threading.Thread(target=self._batch_collector_thread)
        self.batch_thread.start()

    def _worker_thread(self):
        idx = 0
        while not self.stop_event.is_set():
            if idx >= len(self.dataset):
                idx = 0
            img, target = self.dataset.load_image(idx)
            try:
                self.image_queue.put((img, target), timeout=10)
            except queue.Full:
                if self.stop_event.is_set():
                    break
            idx += 1

    def _batch_collector_thread(self):
        while not self.stop_event.is_set():
            images, targets = [], []
            for _ in range(self.batch_size):
                try:
                    img, target = self.image_queue.get(timeout=10)
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                images.append(img)
                targets.append(target)
            images = torch.stack(images)
            targets = torch.tensor(targets)
            try:
                self.batch_queue.put((images, targets), timeout=10)
            except queue.Full:
                if self.stop_event.is_set():
                    break

    def __iter__(self):
        while not self.stop_event.is_set():
            yield self.batch_queue.get(timeout=10)

    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        self.batch_thread.join()
```

The training loop iterates over the Data Loader class which retrieves batches from the batch queue. These are then pushed to the GPU and used for training.

```python    
    # create dataset
    dataset = ImageNetDataset(base_dir, file_list_path, id_to_index, transform=transform, rank=DEFAULT_RANK, world_size=DEFAULT_SIZE)
    
    # user settings
    batch_size = args.nbatch
    num_workers = args.nworkers
    myprint(f'num workers: {num_workers}')
    cache_size = args.cache_size
    total_steps = args.nsteps
    status_print_interval = args.status_print_interval
    profile = args.profile

    data_loader = ImageNetDataLoader(dataset, batch_size, num_workers, cache_size)

    device = torch.device(f'cuda:{DEFAULT_LOCAL_RANK}' if torch.cuda.is_available() else "cpu")

    # Your model definition
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.to(device)
    model = DDP(model, device_ids=[DEFAULT_LOCAL_RANK])

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    # set profile path to be ./logdir/profiler/ + date-time
    log_path = os.path.join('./logdir', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

    step_time = time.time()
    step = 0
    image_rate = MeanCalc()

    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        step += 1
        
        outputs = model(images)
        
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if profile and DEFAULT_RANK==0: prof.step()
        
        if step % status_print_interval == 0 and DEFAULT_RANK==0:
            step_img_rate = status_print_interval * batch_size / (time.time() - step_time)
            myprint(f'step_img_rate: {step_img_rate:.2f}')
            if step > 5:
                image_rate.add(step_img_rate)
            step_time = time.time()
        
        # stop loop
        if step > total_steps:
            break
    
    if profile and DEFAULT_RANK==0: prof.stop()
    if DEFAULT_RANK==0: myprint(f'Average image rate: {str(image_rate)}')

    myprint('Stopping data loader')
    data_loader.stop()
    cleanup()
    myprint(f'All Done: {time.time() - total_start:.2f}')

```

## Results Serial vs Parallel

Using the `submit_polaris.sh` script to run with `PROFILE=<leave-blank>` the output Mean Image Rate for each run is as follows:

```shell
[1] run serial example
Average image rate: mean: 33.01, stddev: 8.68
[53] run parallel with 1 workers
Average image rate: mean: 40.83, stddev: 20.48
[109] run parallel with 2 workers
Average image rate: mean: 506.93, stddev: 195.88
[140] run parallel with 3 workers
Average image rate: mean: 525.26, stddev: 193.35
[170] run parallel with 4 workers
Average image rate: mean: 519.14, stddev: 190.55
```

It can be seen that Serial and 1 Worker are comparable, and adding workers quickly increases throughput. Your results for these measurements will vary because this data pipeline is very I/O inefficient. In a production case, one would run a pre-processing script to put all these images into a combined tensor object and store them in large (10-100GB) files on ALCF filesystems. Reading each JPEG every time it is accessed is terribly inefficient. 

# Using PyTorch Profiler in Tensorboard

The scripts include the ability to profile using the [PyTorch Profiler module](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html). 

This code creates the profiler.

```python
 # create profiler
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=5, warmup=1, active=10, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    )
```

The functions calls `prof.start()`, `prof.step()`, and `prof.stop()` are used in the code to start & stop the profiler and to mark each step of the loop.

## Viewing output

You can view the output by logging into polaris using the following. It is typically best to do this in a secondary terminal after having already logged in.

```bash
ssh -D <choose-a-port> polaris.alcf.anl.gov
```

Then, setup your python environment that you used for running this job and then launch tensorboard:
```shell
module use /soft/modulefiles
module load conda
conda activate

cd ATPESC_MachineLearning/02_dataPipelines/01_pytorchDatasetAPI
tensorboard --logdir ./logdir --bind_all
```

At this point you should see a message like this:
```shell
TensorBoard 2.17.0 at http://polaris-login-02.hsn.cm.polaris.alcf.anl.gov:6006/ (Press CTRL+C to quit)
```

You can open your local web browser, set the Socks5 proxy to `<choose-a-port>`, then navigate to the URL: `localhost:6006`. You should see your profiling runs in the Tensorboard viewer. Be aware that these profiles can be large, 500MB - 1000MB, and complex. Sometimes Tensorboard gets killed by the login-node monitoring services because it exceeds the 8GB application maximum.

