from mpi4py import MPI
import torch


#Ask MPI for rank and total ranks.
rank = int(MPI.COMM_WORLD.Get_rank())
world_size = int(MPI.COMM_WORLD.Get_size())
#Check how many GPUs we have
device_count = torch.cuda.device_count()

#Set us to use the correct GPU
local_rank = rank % device_count
torch.cuda.set_device(local_rank)

# Call the init process
torch.distributed.init_process_group(
    backend="nccl",
    world_size=world_size,
    rank=rank,
)

