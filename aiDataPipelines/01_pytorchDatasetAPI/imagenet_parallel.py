import os,sys

DEFAULT_LOCAL_RANK = int(os.environ.get('PMI_LOCAL_RANK',0))
DEFAULT_LOCAL_SIZE = int(os.environ.get('PMI_LOCAL_SIZE',1))
DEFAULT_RANK = int(os.environ.get('PMI_RANK',0))
DEFAULT_SIZE = int(os.environ.get('PMI_SIZE',1))

print(f'local_rank: {DEFAULT_LOCAL_RANK}, local_size: {DEFAULT_LOCAL_SIZE}, rank: {DEFAULT_RANK}, size: {DEFAULT_SIZE}')

# custom print function that only print on rank 0
def myprint(*args):
    if(DEFAULT_RANK==0): 
        print(*args)
        sys.stdout.flush()
        sys.stderr.flush()

import threading
import queue
import time
import argparse
total_start = time.time()
import datetime
from PIL import Image
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms, models


# simple class to calculate mean and standard deviation
class MeanCalc:
    def __init__(self):
        self.sum = 0
        self.sum2 = 0
        self.n = 0

    def add(self, x):
        self.sum += x
        self.sum2 += x * x
        self.n += 1
    
    def mean(self):
        return self.sum / self.n

    def stddev(self):
        return np.sqrt(self.sum2 / self.n - self.mean()*self.mean())
    
    def __str__(self):
        return f'mean: {self.mean():.2f}, stddev: {self.stddev():.2f}'

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

# the filenames contain a unique id for each image that corresponds to the object ID
# create a hash table for labels from string to int
def build_id_to_index_mapping(file_list_path):
    unique_ids = set()
    with open(file_list_path, 'r') as file:
        for line in file:
            unique_id = line.strip().split('/')[-2]
            unique_ids.add(unique_id)
    return {unique_id: idx for idx, unique_id in enumerate(sorted(unique_ids))}


# DDP: initialize library.
def setup(rank, world_size,backend="nccl"):
    # initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

# DDP: cleanup
def cleanup():
    dist.destroy_process_group()

# transform to resize and convert to tensor
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-w','--nworkers', type=int, default=1)
    parser.add_argument('-b','--nbatch', type=int, default=64)
    parser.add_argument('--cache-size', type=int, default=2)
    parser.add_argument('-s','--nsteps', type=int, default=20)
    parser.add_argument('--profile', action='store_true',default=False)
    parser.add_argument('--base-dir', type=str, default='/lus/eagle/projects/datasets/ImageNet/ILSVRC')
    parser.add_argument('--file-list-path', type=str, default='ilsvrc_train_filelist.txt')
    parser.add_argument('--status-print-interval', type=int, default=5)

    args = parser.parse_args()


    setup(DEFAULT_LOCAL_RANK, DEFAULT_LOCAL_SIZE)
    base_dir = args.base_dir
    file_list_path = os.path.join(base_dir, args.file_list_path)
    id_to_index = build_id_to_index_mapping(file_list_path)

    dataset = ImageNetDataset(base_dir, file_list_path, id_to_index, transform=transform, rank=DEFAULT_RANK, world_size=DEFAULT_SIZE)

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

    # create profiler
    prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=20, warmup=1, active=20, repeat=1),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    )

    if profile and DEFAULT_RANK==0: prof.start()

    step_time = time.time()
    step = 0
    image_rate = MeanCalc()

    myprint('Starting data loader')
    for images, targets in data_loader:
        images = images.to(device)
        targets = targets.to(device)
        step += 1
        # myprint(f'step: {step}')
        
        outputs = model(images)
        
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if profile and DEFAULT_RANK==0: prof.step()
        
        if step % status_print_interval == 0 and DEFAULT_RANK==0:
            step_img_rate = status_print_interval * batch_size / (time.time() - step_time)
            myprint(f'step: {step}; step_img_rate: {step_img_rate:.2f}')
            if step > 5:
                image_rate.add(step_img_rate)
            step_time = time.time()
        
        # stop loop
        if step > total_steps:
            break
    
    if profile and DEFAULT_RANK==0:
        prof.stop()
        prof.export_chrome_trace(f'imagenet_parallel-nw{num_workers}.json')
    if DEFAULT_RANK==0: myprint(f'Average image rate: {str(image_rate)}')

    myprint('Stopping data loader')
    data_loader.stop()
    cleanup()
    myprint(f'All Done: {time.time() - total_start:.2f}')

if __name__ == '__main__':
    main()