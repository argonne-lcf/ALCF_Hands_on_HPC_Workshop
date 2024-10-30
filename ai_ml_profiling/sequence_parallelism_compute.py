"""
Partial benchmark focusing on the compute and communication pattern of
sequence parallelism.

Note: Here all the operations overwrite the input tensor. The input is, defined
as the input per rank. The matrix shapes are chosen to mimic a particular 
implementation of sequence parallelism in Megatron-DeepSpeed. 

Important Variables:
    - d1, S: sequence length
    - d2, H: number of hidden dimension
    - precision type: float32
    - mm1: weight matrix to be applied before the attention layer
    - mm2: weight matrix to be applied after the attention layer
    - time0: total time taken for the first all_gather in the timing loop
    - time1: total time taken for the application of the first weight matrix
    - time2: total time taken for the application of the second wight matrix
    - time3: total time taken for the reduce scatter operation
    - inmean0: mean value of the inputs
    - inmean2: mean value of the results

Pattern of compute operations:
    - Intialize data on available ranks on a node
    - Perform an all_gather
    - Apply weight matrix
    - Apply attention matrix (to be implemented or come from DeepSpeed)
    - Apply weight matrix after the attention
    - Perform a reduce scatter of the result

Data and Input shapes:
    - All Gather Buffer (A) shape: (S, 1, H) == (d1, 1, d2)
    - Input per rank: (S/NGPUS, 1, H) == (d1//12, 1, d2)
    - Weight matrix 1, mm1: (H//12, H) == (d2//12, d1)
    - Weight matrix 2, mm2: (H, H//12) == (d1, d2//12)
    - Result after application of mm1, A*mm1^T == (S, 1, H//12) == (d1, 1, d2//12)
    - Result after application of mm2, A*mm1^T*mm2^T == (S, 1, H) == (d1, 1, d2)
"""
#print("being import", flush=True)
from mpi4py import MPI
import os
import time
import socket
import argparse

import torch

#import intel_extension_for_pytorch as ipex  # noqa: F401 # type: ignore
#import oneccl_bindings_for_pytorch  # noqa: F401 # type: ignore
#print("being code", flush=True)

parser = argparse.ArgumentParser(description="parse input arguments for sequence parallel partial benchmark")

parser.add_argument("-s", "--sequence_length", 
                    help="Maximum sequence length. The size of the ALLGATHER buffer", 
                    type=int, default=4608)
parser.add_argument("-d", "--hidden_dimension", 
                    help="Hidden dimension for the matrix multiplication. Proxy for the model size.", 
                    type=int, default=9216)
parser.add_argument("-it", "--iterations", 
                    help="number of iterations for the timing loop", 
                    type=int, default=18)
parser.add_argument("-wit", "--warmup_iterations", help="number of warmup iterations", 
                    type=int, default=10)
parser.add_argument("-p", "--precision", help="Data type for the elements of a tensor.", 
                    type=str, default="float32")

args = parser.parse_args()

rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())
print(f"rank {rank}/{world_size}")
device_count = torch.cuda.device_count()
#device_count = int(os.environ["NGPU_PER_HOST"])

device = rank % device_count
local_rank = device
os.environ['CCL_LOCAL_RANK'] = str(device)
os.environ['CCL_LOCAL_SIZE'] = str(device_count)
backend = "nccl"

if rank == 0:
   master_addr              = socket.gethostname()
   sock                     = socket.socket()
   sock.bind(('',0))
   # master_port  = sock.getsockname()[1]
   master_port              = 2345
else:
   master_addr              = None
   master_port              = None

master_addr                 = MPI.COMM_WORLD.bcast(master_addr, root=0)
master_port                 = MPI.COMM_WORLD.bcast(master_port, root=0)
os.environ["MASTER_ADDR"]   = master_addr
os.environ["MASTER_PORT"]   = str(master_port)

torch.cuda.set_device(device)
torch.distributed.init_process_group(
    backend=backend,
    init_method="env://",
    world_size=world_size,
    rank=rank,
)

if args.precision == "float32":
    data_type = torch.float32
    data_type_multiplier = 32 ## 32 Bits = 4 Bytes
elif args.precision == "bfloat16":
    data_type = torch.bfloat16
    data_type_multiplier = 16 ## 16 Bits

d1 = args.sequence_length #4608 #4608 sequence length
d2 = args.hidden_dimension #9216 #9216 hidden dimension
all_gather_buffer = torch.zeros([d1, 1, d2], dtype=data_type, device=f"cuda:{torch.cuda.current_device()}")
input = torch.rand([d1//world_size, 1, d2], dtype=data_type, device=f"cuda:{torch.cuda.current_device()}")
#print(f"Input shape = {input.shape}")
mm1 = torch.rand(
        d2//world_size,
        d2,
        device=f"cuda:{torch.cuda.current_device()}",
        dtype=data_type,
    )*1e-10
mm2 = torch.rand(
        d2,
        d2//world_size,
        device=f"cuda:{torch.cuda.current_device()}",
        dtype=data_type,
    )*1e-8
#warmup
input_mean_before_operations=input.mean()
#print("in_mean0", input.mean())
for i in range(args.warmup_iterations):
    torch.distributed.all_gather_into_tensor(
        all_gather_buffer, input
    )
    intermediate = torch.matmul(all_gather_buffer, mm1.t())
    #FA here
    intermediate = torch.matmul(intermediate, mm2.t())
    torch.distributed.reduce_scatter_tensor(
        input, intermediate
    )
print("start loop", flush=True)
#print("in_mean1", input.mean())
time0 = 0.0
time1 = 0.0
time2 = 0.0
time3 = 0.0

list_all_gather_times = []
list_reduce_scatter_times = []

start_time=time.time()
for i in range(args.iterations):
    start = time.time()
    torch.distributed.all_gather_into_tensor(
        all_gather_buffer, input
    )
    torch.cuda.synchronize()
    end = time.time()
    time0 += end-start
    list_all_gather_times.append(end-start)
    start = end

    intermediate = torch.matmul(all_gather_buffer, mm1.t())
    torch.cuda.synchronize()
    end = time.time()
    time1 += end-start
    #FA would be here
    start = end
    intermediate = torch.matmul(intermediate, mm2.t())
    torch.cuda.synchronize()
    end = time.time()
    time2 += end-start
    start = end
    torch.distributed.reduce_scatter_tensor(
        input, intermediate
    )
    torch.cuda.synchronize()
    end = time.time()
    time3 += end-start
    list_reduce_scatter_times.append(end-start)
    #gather optimizer states
    #allreduce model updates
end_time=time.time()
torch.cuda.synchronize()
if rank == 0:
    print(f"Running with {args.precision} data type")
    print(f"\n ==== List of Arguments ==== \n")
    print(f"Sequence Length = {args.sequence_length}")
    print(f"Hidden Dimension = {args.hidden_dimension}")
    print(f"Precision Type = {args.precision}")
    print("\n ==== List of Arguments ==== \n")
    print(f"Input shape = {input.shape}")
    print(f"Matrix 1 shape = {mm1.shape}")
    print(f"Matrix 2 shape = {mm2.shape}")
    print(f"ALLGATHER Buffer size = {(args.sequence_length * data_type_multiplier) / 8 / 1000 / 1000} MB")
    #print("times", time0*1000, time1*1000, time2*1000, time3*1000)
    print(f"Mean before all operations = {input_mean_before_operations}")
    print(f"Total time taken for ALLGATHER = {time0*1000} ms" )
    print(f"Total time taken for matrix multiplication 1 = {time1*1000} ms")
    print(f"Total time taken for matrix multiplication 2 = {time2*1000} ms")
    print(f"Total time taken for REDUCE_SCATTER = {time3*1000} ms")
    print(f"total time for the loop = {(end_time-start_time)*1000} ms")
    print("Mean after all operations = ", input.mean())
    print(f"ALLGATHER Throughput = {(args.sequence_length * data_type_multiplier * args.iterations) / (time0 * 1000 * 1000) / 1024 / 1024 } MB/s")
    print(f"REDUCE_SCATTER Throughput = {(args.sequence_length * data_type_multiplier * args.iterations) / (time3 * 1000 * 1000) / 1024 / 1024 } MB/s")

    for idx, (t_ag, t_rs) in enumerate(zip(list_all_gather_times, list_reduce_scatter_times)):
        print(f"ALLGATHER {idx} takes {t_ag*1000} ms, REDUCE_SCATTER {idx} takes {t_rs*1000} ms")
