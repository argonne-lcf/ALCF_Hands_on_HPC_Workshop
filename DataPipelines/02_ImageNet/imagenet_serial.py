# This example builds a serial data pipeline to use as an example
import os
import time
import argparse
import datetime
import time
total_start = time.time()
from PIL import Image
import numpy as np
import torch
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
   def __init__(self, base_dir, file_list_path, id_to_index, transform=None):
      self.base_dir = base_dir
      self.transform = transform
      self.id_to_index = id_to_index
      with open(file_list_path, 'r') as file:
         self.image_paths = [line.strip() for line in file]

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

# the filenames contain a unique id for each image that corresponds to the object ID
# create a hash table for labels from string to int
def build_id_to_index_mapping(file_list_path):
   unique_ids = set()
   with open(file_list_path, 'r') as file:
      for line in file:
         unique_id = line.strip().split('/')[-2]
         unique_ids.add(unique_id)
   return {unique_id: idx for idx, unique_id in enumerate(sorted(unique_ids))}

# transform to resize and convert to tensor
transform = transforms.Compose([
   transforms.Resize((256, 256)),
   transforms.ToTensor(),
])

def main():

   # parse arguments
   parser = argparse.ArgumentParser()
   parser.add_argument('-b','--nbatch', type=int, default=64)
   parser.add_argument('-s','--nsteps', type=int, default=20)
   parser.add_argument('--profile', action='store_true',default=False)
   parser.add_argument('--base-dir', type=str, default='/lus/eagle/projects/datasets/ImageNet/ILSVRC')
   parser.add_argument('--file-list-path', type=str, default='ilsvrc_train_filelist.txt')
   parser.add_argument('--status-print-interval', type=int, default=5)
   args = parser.parse_args()

   # setup dataset 
   base_dir = args.base_dir
   file_list_path = os.path.join(base_dir, args.file_list_path)
   id_to_index = build_id_to_index_mapping(file_list_path)
   dataset = ImageNetDataset(base_dir, file_list_path, id_to_index, transform=transform)

   # run settings
   batch_size = args.nbatch
   print(f'Batch size: {batch_size}')
   total_steps = args.nsteps
   status_print_interval = args.status_print_interval
   profile = args.profile

   # set device to gpu if available
   device = torch.device(f'cuda:0' if torch.cuda.is_available() else "cpu")

   # Create a model to use as an example
   model = models.resnet50(weights='IMAGENET1K_V1')
   model.to(device)
   criterion = torch.nn.CrossEntropyLoss().to(device)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
   model.train()

   # set profile path to be ./logdir/profiler/ + date-time
   log_path = os.path.join('./logdir', datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))

   # create pytorch profiler that outputs TensorBoard logs
   prof = torch.profiler.profile(
         schedule=torch.profiler.schedule(wait=20, warmup=1, active=20, repeat=1),
         activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
         # on_trace_ready=torch.profiler.tensorboard_trace_handler(log_path),
         record_shapes=True,
         profile_memory=True,
         with_stack=True
   )

   # start the profiler
   if profile: prof.start()

   step_time = time.time()
   step = 0
   image_rate = MeanCalc()

   # training loop
   while step < total_steps:
      images, targets = [], []
      # build an input batch serially
      for _ in range(batch_size):
         if step * batch_size + _ >= len(dataset):
            break
         img, target = dataset.load_image(step * batch_size + _)
         images.append(img)
         targets.append(target)
      if len(images) == 0:
         break
      # convert to pytorch tensors
      images = torch.stack(images).to(device)
      targets = torch.tensor(targets).to(device)
      step += 1
      
      # pass the batch through the model
      outputs = model(images)
      loss = criterion(outputs, targets)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if profile: prof.step()
      
      # print status
      if step % status_print_interval == 0:
         step_img_rate = status_print_interval * batch_size / (time.time() - step_time)
         print(f'step: {step}; step_img_rate: {step_img_rate:.2f}')
         if step > 5:
            image_rate.add(step_img_rate)
         step_time = time.time()
   
   if profile: prof.stop()
   if profile: prof.export_chrome_trace("imagenet_serial.json")
   print(f'Average image rate: {str(image_rate)}')

   print(f'All Done; total runtime: {time.time() - total_start:.2f}')

if __name__ == '__main__':
   main()
