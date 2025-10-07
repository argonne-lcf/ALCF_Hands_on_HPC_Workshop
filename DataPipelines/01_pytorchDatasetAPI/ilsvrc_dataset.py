#!/bin/env python

# Updated ImageNet Dataset processing in PyTorch using IterableDataset
# This version removes the index mapping from __init__() and uses IterableDataset
# to process data on-the-fly during iteration.
# Added support for profiling with PyTorch Profiler when using --profile argument.
# Profiling includes GPU activities and names are added to function calls.

import os
import glob
import sys

DEFAULT_LOCAL_RANK = int(os.environ.get('PMI_LOCAL_RANK', 0))
DEFAULT_LOCAL_SIZE = int(os.environ.get('PMI_LOCAL_SIZE', 1))
DEFAULT_RANK = int(os.environ.get('PMI_RANK', 0))
DEFAULT_SIZE = int(os.environ.get('PMI_SIZE', 1))

# Set default rank and world size
if 'RANK' not in os.environ:
    os.environ['RANK'] = os.environ.get('PMI_RANK', '0')
if 'WORLD_SIZE' not in os.environ:
    os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', '1')

# Set master address and port
if 'MASTER_ADDR' not in os.environ:
    os.environ['MASTER_ADDR'] = 'localhost'
if 'MASTER_PORT' not in os.environ:
    os.environ['MASTER_PORT'] = '12399'

print(f'local_rank: {DEFAULT_LOCAL_RANK}, local_size: {DEFAULT_LOCAL_SIZE}, rank: {DEFAULT_RANK}, size: {DEFAULT_SIZE}')
sys.stdout.flush()
sys.stderr.flush()

def myprint(*args):
    if(int(os.environ.get('RANK', 0)) == 0):
        print(*args)
        sys.stdout.flush()
        sys.stderr.flush()

import numpy as np
import xml.etree.ElementTree as ET

# Must be imported after the environment variables are set
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torchvision import io
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.data import get_worker_info

import argparse
import json
import time
from datetime import datetime

# Global variables initialized in get_datasets
labels_hash = None
crop_size = None

def get_datasets(config):
    # These global variables will be initialized
    global labels_hash, crop_size

    # Set the crop size of the output images, e.g., [256, 256]
    crop_size = config['data']['crop_image_size']
    # Paths to text files containing a list of all the training and testing JPEGs
    train_filelist = config['data']['train_filelist']
    test_filelist = config['data']['test_filelist']

    assert os.path.exists(train_filelist)
    assert os.path.exists(test_filelist)

    # Build label mapping
    labels_hash = get_label_tables(train_filelist)

    # Create Dataset objects
    train_ds = ImageNetIterableDataset(config, train_filelist, labels_hash, train=True)
    valid_ds = ImageNetIterableDataset(config, test_filelist, labels_hash, train=False)

    return train_ds, valid_ds

def get_label_tables(train_filelist):
    # Get the first filename
    with open(train_filelist) as file:
        filepath = file.readline().strip()

    # Extract the path up to: /.../ILSVRC/Data/CLS-LOC/train/
    label_path = '/'.join(filepath.split('/')[:-2])
    labels = glob.glob(label_path + os.path.sep + '*')
    if dist.get_rank() == 0:
        print(f'num labels: {len(labels)}')
    # Remove the leading path from the label directories
    labels = [os.path.basename(i) for i in labels]
    labels.sort()  # Ensure consistent ordering
    # Create a mapping from label strings to integers
    labels_hash = {label: idx for idx, label in enumerate(labels)}
    return labels_hash

class ImageNetIterableDataset(IterableDataset):
    def __init__(self, config, filelist_filename, labels_hash, train=True):
        self.config = config
        self.train = train
        self.labels_hash = labels_hash
        dc = config['data']
        self.resize_shape = dc['crop_image_size']

        # Load the full filelist
        with open(filelist_filename) as file:
            self.filelist = [line.strip() for line in file]

        # Provide user with estimated batches per rank
        numranks = dist.get_world_size()
        batches_per_rank = int(len(self.filelist) / dc['batch_size'] / numranks)
        if dist.get_rank() == 0:
            print(f'input filelist contains {len(self.filelist)} files, estimated batches per rank {batches_per_rank}')

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(dc['crop_image_size']),
            transforms.ToTensor(),
            # Add any other transforms here
        ])

        # Manually handle sharding since DistributedSampler doesn't support IterableDataset
        self.numranks = numranks
        self.rank = dist.get_rank()

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            # Single-process data loading, no need to split dataset
            iter_start = 0
            iter_end = len(self.filelist)
            num_workers = 1
            worker_id = 0
        else:
            # Multi-process data loading, split workload
            per_worker = int(np.ceil(len(self.filelist) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.filelist))
            num_workers = worker_info.num_workers

        # Further split the data among multiple processes
        per_rank = int(np.ceil((iter_end - iter_start) / float(self.numranks)))
        rank_start = iter_start + self.rank * per_rank
        rank_end = min(rank_start + per_rank, iter_end)

        for idx in range(rank_start, rank_end):
            image_path = self.filelist[idx]
            label_str = os.path.basename(os.path.dirname(image_path))
            label = self.labels_hash[label_str]
            annot_path = image_path.replace('Data', 'Annotations')
            annot_path = annot_path.replace('JPEG', 'xml')

            # Get all bounding boxes for the image
            with torch.profiler.record_function("get_bounding_boxes"):
                bounding_boxes = get_bounding_boxes(annot_path)

            # Open image
            try:
                with torch.profiler.record_function("open_image"):
                    img = io.read_image(image_path, io.ImageReadMode.RGB)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                continue

            _, width, height = img.shape

            for bbox in bounding_boxes:
                # Crop image according to bbox
                with torch.profiler.record_function("crop_and_transform_image"):
                    xmin = int(bbox[1] * width )
                    ymin = int(bbox[0] * height)
                    xmax = int(bbox[3] * width )
                    ymax = int(bbox[2] * height)

                    cropped_img = img[:, xmin:xmax, ymin:ymax]

                    # Apply transforms
                    img_tensor = transforms.Resize(self.resize_shape)(cropped_img)

                # Yield the sample
                yield img_tensor, label

def get_bounding_boxes(filename):
    try:
        tree = ET.parse(filename)
        root = tree.getroot()

        img_size = root.find('size')
        img_width = int(img_size.find('width').text)
        img_height = int(img_size.find('height').text)

        objs = root.findall('object')
        bndbxs = []
        for obj in objs:
            bndbox = obj.find('bndbox')
            bndbxs.append([
                float(bndbox.find('ymin').text) / (img_height - 1),
                float(bndbox.find('xmin').text) / (img_width - 1),
                float(bndbox.find('ymax').text) / (img_height - 1),
                float(bndbox.find('xmax').text) / (img_width - 1)
            ])
    except FileNotFoundError:
        bndbxs = [[0, 0, 1, 1]]
    except Exception as e:
        print(f"Error parsing annotation {filename}: {e}")
        bndbxs = [[0, 0, 1, 1]]

    return bndbxs

if __name__ == '__main__':
    # Parse command line
    import argparse
    import json
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Data Pipeline Example with DDP and IterableDataset')
    parser.add_argument('-c', '--config', dest='config_filename',
                        help='configuration filename in json format',
                        required=True)
    parser.add_argument('-l', '--logdir', dest='logdir',
                        help='log output directory', default='logdir')
    parser.add_argument('-n', '--nsteps', dest='nsteps',
                        help='number of steps to run', default=10, type=int)
    parser.add_argument('--num-workers', type=int, help='number of DataLoader workers', default=4)
    parser.add_argument('--batch-size', type=int, help='batch size', default=None)
    parser.add_argument('--local-rank', type=int, help='local rank for distributed training', default=DEFAULT_LOCAL_RANK)
    parser.add_argument('--profile', type=str, help='Path to save profiler output')

    args = parser.parse_args()

    # Initialize process group for distributed training
    dist.init_process_group(backend='nccl')

    # Set up device
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cpu')

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    myprint(f"Rank: {rank}, World Size: {world_size}")

    # Parse config file
    config = json.load(open(args.config_filename))

    if args.batch_size:
        config['data']['batch_size'] = args.batch_size

    # Call function to build dataset objects
    # Both of the returned objects are PyTorch Dataset objects
    train_ds, test_ds = get_datasets(config)

    # Since DistributedSampler doesn't support IterableDataset, we don't use it
    train_loader = DataLoader(
        train_ds,
        batch_size=config['data']['batch_size'],
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Iterate over the DataLoader object
    start = time.time()
    nsteps = args.nsteps
    myprint(f'starting loop with nsteps = {nsteps} and rank = {rank} and num_workers = {args.num_workers}')

    step = 0

    if args.profile:
        from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler

        profile_dir = os.path.join(args.profile, f'rank_{rank}')
        if not os.path.exists(profile_dir):
            os.makedirs(profile_dir)

        # Set up the profiler
        with profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=nsteps, repeat=1),
            on_trace_ready=tensorboard_trace_handler(profile_dir),
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as profiler:
            for inputs, labels in train_loader:
                if step >= nsteps + 3:
                    break
                with torch.profiler.record_function("data_transfer_to_device"):
                    inputs, labels = inputs.to(device), labels.to(device)
                myprint(f'batch_number = {step} input device = {inputs.device} labels device = {labels.device}')
                profiler.step()
                step += 1
    else:
        for inputs, labels in train_loader:
            if step >= nsteps:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            myprint(f'batch_number = {step} input device = {inputs.device} labels device = {labels.device}')
            step += 1

    # Measure performance in images per second
    duration = time.time() - start
    images = config['data']['batch_size'] * nsteps
    myprint('imgs/sec = %5.2f' % ((images / duration) * world_size))
    dist.destroy_process_group()
    myprint('done')
    os._exit(0)
    print('after')
