"""
main.py

Conains simple implementation illustrating how to use PyTorch DDP for
distributed data parallel training.
"""
from __future__ import absolute_import, annotations, division, print_function
import os
import socket
import time
from typing import Callable, Optional, Union

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import wandb
from hplib.configs import PROJECT_DIR
# from torch.nn.parallel import DistributedDataParallel as DDP


from hplib.utils.pylogger import get_pylogger

# log = logging.getLogger(__name__)
log = get_pylogger(__name__)
#
# wblog = logging.getLogger("wandb")
# wblog.setLevel(logging.WARNING)



# Get MPI:
try:
    from mpi4py import MPI
    WITH_DDP = True
    LOCAL_RANK = int(
        os.environ.get(
            'PMI_LOCAL_RANK',
            os.environ.get(
                'OMPI_COMM_WORLD_LOCAL_RANK',
                '0'
            )
        )
    )
    # LOCAL_RANK = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    SIZE = MPI.COMM_WORLD.Get_size()
    RANK = MPI.COMM_WORLD.Get_rank()

    WITH_CUDA = torch.cuda.is_available()
    DEVICE = 'gpu' if WITH_CUDA else 'CPU'

    # pytorch will look for these
    os.environ['RANK'] = str(RANK)
    os.environ['WORLD_SIZE'] = str(SIZE)
    # -----------------------------------------------------------
    # NOTE: Get the hostname of the master node, and broadcast
    # it to all other nodes It will want the master address too,
    # which we'll broadcast:
    # -----------------------------------------------------------
    MASTER_ADDR = socket.gethostname() if RANK == 0 else None
    MASTER_ADDR = MPI.COMM_WORLD.bcast(MASTER_ADDR, root=0)
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    os.environ['MASTER_PORT'] = str(2345)
    # log = logging.getLogger(__name__)
    # log.setLevel(logging.DEBUG)
    # if RANK != 0 or LOCAL_RANK != 0:
    #     log.setLevel(logging.CRITICAL)

except (ImportError, ModuleNotFoundError) as e:
    WITH_DDP = False
    SIZE = 1
    RANK = 0
    LOCAL_RANK = 0
    MASTER_ADDR = 'localhost'
    log.warning('MPI Initialization failed!')
    log.warning(e)


Tensor = torch.Tensor


def init_process_group(
    rank: Union[int, str],
    world_size: Union[int, str],
    backend: Optional[str] = None,
) -> None:
    if WITH_CUDA:
        backend = 'nccl' if backend is None else str(backend)

    else:
        backend = 'gloo' if backend is None else str(backend)

    dist.init_process_group(
        backend,
        rank=int(rank),
        world_size=int(world_size),
        init_method='env://',
    )


def cleanup():
    dist.destroy_process_group()


def metric_average(val: Tensor):
    if (WITH_DDP):
        # Sum everything and divide by the total size
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        return val / SIZE

    return val


def run_demo(demo_fn: Callable, world_size: int | str) -> None:
    mp.spawn(demo_fn,  # type: ignore
             args=(world_size,),
             nprocs=int(world_size),
             join=True)


def train_mnist(cfg: DictConfig, wbrun: Optional[wandb.run] = None) -> float:
    from hplib.trainer import Trainer
    start = time.time()
    tconfig = instantiate(cfg.get('trainer'))
    net_config = instantiate(cfg.get('network'))
    trainer = Trainer(config=tconfig, net_config=net_config, wbrun=wbrun)
    epoch_times = []
    for epoch in range(1, tconfig.epochs + 1):
        t0 = time.time()
        metrics = trainer.train_epoch(epoch)
        epoch_times.append(time.time() - t0)

        if epoch % tconfig.logfreq and RANK == 0:
            acc = trainer.test()
            astr = f'[TEST] Accuracy: {100.0 * acc:.0f}%'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={metrics["loss"]:.4f}',
                f'acc={metrics["acc"] * 100.0:.0f}%'
            ])
            if wbrun is not None:
                wbrun.log(
                    {
                        'epoch/epoch': epoch,
                        **{f'epoch/{k}': v for k, v in metrics.items()}
                    }
                )
            log.info((sep := '-' * len(summary)))
            log.info(summary)
            log.info(sep)

    rstr = f'[{RANK}] ::'
    if RANK == 0:
        log.info(' '.join([
            rstr,
            f'Total training time: {time.time() - start} seconds'
        ]))
        avg_over = min(5, (len(epoch_times) - 1))
        avg_epoch_time = np.mean(epoch_times[-avg_over:])
        log.info(' '.join([
            rstr,
            f'Average time per epoch in the last {avg_over}: {avg_epoch_time}'

        ]))

    test_acc = trainer.test()
    return test_acc


def train_epoch_and_eval(cfg: DictConfig, wbrun: Optional[wandb.run] = None) -> float:
    from hplib.trainer import Trainer
    tconfig = instantiate(cfg.get('trainer'))
    net_config = instantiate(cfg.get('network'))
    trainer = Trainer(config=tconfig, net_config=net_config, wbrun=wbrun)
    t0 = time.time()
    _ = trainer.train(0)
    log.info(f'Training took: {time.time() - t0:.4f}s')
    return trainer.test()


def setup_wandb(cfg: DictConfig) -> dict:
    wbrun = None
    if RANK == 0 and LOCAL_RANK == 0:
        wbcfg = cfg.get('wandb', None)
        if wbcfg is not None:
            import wandb
            from wandb.util import generate_id
            run_id = generate_id()
            wbrun = wandb.init(
                dir=os.getcwd(),
                id=run_id,
                mode='online',
                resume='allow',
                # magic=True,
                # save_code=True,
                project='sdl-wandb-test',
                # entity='',  # wbcfg.setup.entity,
            )
            wbrun.log_code(cfg.get('work_dir', PROJECT_DIR))
            wbrun.config.update(OmegaConf.to_container(cfg, resolve=True))
            wbrun.config['run_id'] = run_id
            wbrun.config['logdir'] = os.getcwd()
            wbrun.config['NRANKS'] = SIZE  # os.environ.get('NRANKS', SIZE)
            wbrun.config['hostname'] = MASTER_ADDR
            wbrun.config['device'] = (
                'gpu' if torch.cuda.is_available() else 'cpu'
            )

    return {'run': wbrun}



def run(cfg: DictConfig) -> float:
    wb = setup_wandb(cfg)
    backend = 'NCCL' if torch.cuda.is_available() else 'gloo'
    init_process_group(RANK, world_size=SIZE, backend=backend)
    # train_mnist(cfg, wbrun)
    # run(cfg, wbrun)
    test_acc = train_mnist(cfg, wb['run'])
    return test_acc


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    run(cfg)
    cleanup()


if __name__ == '__main__':
    wandb.require('service')
    main()
