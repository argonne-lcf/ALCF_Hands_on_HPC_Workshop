import datetime
from time import perf_counter_ns
import sys
import os
import socket
from mpi4py import MPI
import argparse

import logging

parser = argparse.ArgumentParser(description="parse input arguments for the gpu allreduce benchmark")

parser.add_argument("-dim", "--tensor_dimension_1d",
                        help="The size of the 1d tensor that is distributed accross the ranks per node.",
                        type=int, default=1073741824) ## ~2.15 GB per rank, in BF16
parser.add_argument("-p", "--precision", help="Data type for the elements of a tensor. float32 and bfloat16 supported.",
                    type=str, default="bfloat16")
args = parser.parse_args()

logging.basicConfig(level="INFO")

def main(tensor_dimension_1d):
    t1 = perf_counter_ns() 
    import intel_extension_for_pytorch  # Added Extra
    import torch
    import torch.nn.parallel
    import torch.distributed as dist
    import oneccl_bindings_for_pytorch
    t2 = perf_counter_ns() 
    import_timer = t2 - t1

    if args.precision == "float32":
        data_type = torch.float32
        data_type_multiplier = 4 ## 32 Bits = 4 Bytes
    elif args.precision == "bfloat16":
        data_type = torch.bfloat16
        data_type_multiplier = 2 ## 16 Bits = 2 Bytes

    MPI.COMM_WORLD.Barrier()

    os.environ['RANK']          = str(os.environ.get('PMI_RANK', 0))
    os.environ['WORLD_SIZE']    = str(os.environ.get('PMI_SIZE', 1))
    mpi_world_size              = MPI.COMM_WORLD.Get_size()
    mpi_my_rank                 = MPI.COMM_WORLD.Get_rank()

    if mpi_my_rank == 0:
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

    MPI.COMM_WORLD.Barrier()
    t3 = perf_counter_ns() 
    dist.init_process_group(backend = "ccl", init_method = 'env://', world_size = mpi_world_size, rank = mpi_my_rank, timeout = datetime.timedelta(seconds=3600))
    t4 = perf_counter_ns() 
    init_timer = t4 - t3
    MPI.COMM_WORLD.Barrier()


    dist_my_rank        = dist.get_rank()
    dist_world_size     = dist.get_world_size()
    device_count = torch.xpu.device_count()

    def get_default_device():
        if torch.xpu.is_available():
            #return torch.device(f"cuda:{dist_my_rank%4}")
            return torch.device(f"xpu:{dist_my_rank%int(device_count)}")
        else:
            return torch.device('cpu')

    device  = get_default_device()
    torch.xpu.set_device(device)

    #dim_size=int(int(sys.argv[1])/4)
    #dim_size=int(int(tensor_dimension_1d)/4)
    dim_size=int(tensor_dimension_1d)

    MPI.COMM_WORLD.Barrier()

    elapsed1=[]
    total_elapsed=0.0

    for _ in range(10):
        x = torch.ones([1, dim_size],dtype=data_type).to(device, non_blocking=True)
        # print(x)
        t5 = perf_counter_ns() 
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # Added Extra op
        #MPI.COMM_WORLD.Barrier()
        torch.xpu.synchronize()
        t6 = perf_counter_ns()
        elapsed1.append(t6 - t5)
        total_elapsed += (t6-t5)

    if mpi_my_rank == 0:
        logging.info(f"Python Import time = {import_timer / 1000 / 1000 / 1000} s")
        logging.info(f"DDP initialization time = {init_timer / 1000 / 1000 / 1000} s")
        logging.info(f"Precision Type: {data_type}")
        logging.info(f"Message size = {(dim_size * data_type_multiplier) / 1000 / 1000} MB")
        logging.info(f"Total time = {total_elapsed / 1000 / 1000 / 1000} s")
        for idx, e in enumerate(elapsed1):
            if idx==0:
                logging.info(f"ALLREDUCE {idx} took {e / 1000 / 1000 / 1000} s, Throughput = {((dim_size * data_type_multiplier) / e) * 1000} MB/s")
            else:
                logging.info(f"ALLREDUCE {idx} took {e / 1000 / 1000} ms, Throughput = {((dim_size * data_type_multiplier) / e) * 1000} MB/s")

if __name__ == "__main__":    
    main(args.tensor_dimension_1d)



