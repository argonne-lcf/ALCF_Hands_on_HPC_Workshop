# Large Language Models on Aurora

Sam Foreman  
_2025-05-06_

> [!NOTE]
> This is a markdown version of the original slides.
> The original slides can be found at:
> - HTML Version: [samforeman.me/talks/incite-hackathon-2025](https://samforeman.me/talks/incite-hackathon-2025)
> - Reveal.js (slides) Version: [samforeman.me/talks/incite-hackathon-2025/slides](https://samforeman.me/talks/incite-hackathon-2025/slides)

## Contents

- [📍 Currently](#-currently)
- [💬 LLMs on Aurora](#-llms-on-aurora)
- [🍋 `ezpz`](#-ezpz)
- [🐣 Getting Started](#-getting-started)
- [🏖️ Shell Environment](#-shell-environment)
- [🔍 Environment Setup with `ezpz_setup_env`](#-environment-setup-with-ezpz_setup_env)
- [⏱️ Working with Job Scheduler(s)](#%EF%B8%8F-working-with-job-schedulers)
- [🔄 Use Custom Node Lists](#-use-custom-node-lists)
- [🐍 Python Environments](#-python-environments)
- [📦 Install `ezpz`](#-install-ezpz)
- [➕ How to Modify Existing
  Code](#-how-to-modify-existing-code)
- [✨ Features](#-features)
- [🧪 Experiment Tracking](#-experiment-tracking)
- [🤏 Minimal Example](#-minimal-example)
- [🏃‍♂️ Running the Minimal Example](#%EF%B8%8F-running-the-minimal-example) 
- [📝 `ezpz-test`](#-ezpz-test)
- [🦜 Generate Text](#-generate-text)
- [🤗 Huggingface Trainer](#-huggingface-trainer)
- [🏎️ Megatron-DeepSpeed](#%EF%B8%8F-megatron-deepspeed)
- [🙌 Acknowledgements](#-acknowledgements)

## 📍 Currently

<div id="fig-pretraining-infrastructure">


<img width="66%" alt="Modern Pretraining Infrastructure" src="https://pbs.twimg.com/media/GqEG2yxXUAAN5kp?format=png&name=small">

Figure 1: Current state of LLM Pretraining.
\[[Source](https://x.com/Dorialexander/status/1918822518804132085)\]

</div>

## 💬 LLMs on Aurora

- 🍋 [`ezpz`](https://github.com/saforem2/ezpz)
- 🤗 [`transformers`](https://github.com/huggingface/transformers)
- 🏎️
  [`Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)

## 🍋 `ezpz`

> Write once, run anywhere

## 🐣 Getting Started

1.  Submit interactive job:

    ``` bash
    qsub -I -l select=2 -l walltime=01:00:00 \
        -l filesystems=home:flare \
        -A gpu_hack \
        -q gpu_hack_prio
    ```

2.  Source[^1] the
    [`ezpz/bin/utils.sh`](https://github.com/saforem2/ezpz/blob/main/bin/utils.sh)
    script (using `curl` to download it[^2]):

    ``` bash
    source <(curl -L https://bit.ly/ezpz-utils)
    ```

## 🏖️ Shell Environment

1.  Setup environment:

    ``` bash
    ezpz_setup_env
    ```

    - <details closed>

      <summary>

      Output:
      </summary>

      ``` bash
      ; source <(curl -L https://bit.ly/ezpz-utils) && ezpz_setup_env
      [2025-05-05-072645][W] PBS_O_WORKDIR is not set! Setting it to current working directory
      [2025-05-05-072645][I] Exporting PBS_O_WORKDIR=/lus/flare/projects/datascience/foremans/projects/saforem2/ezpz
      [2025-05-05-072645][I]  ===== Running Full Environment Setup =====
      [2025-05-05-072645][I] [PYTHON]
      [2025-05-05-072645][I]   - No conda_prefix OR virtual_env found in environment. Setting up conda...
      [2025-05-05-072645][I] Setting up conda on aurora
      [2025-05-05-072647][I] List of active modules:

      Currently Loaded Modules:
          1) gcc-runtime/13.3.0-ghotoln (H)   7) libiconv/1.17-jjpb4sl         (H)  13) cray-pals/1.4.0
          2) gmp/6.3.0-mtokfaw          (H)   8) libxml2/2.13.5                     14) cray-libpals/1.4.0
          3) mpfr/4.2.1-gkcdl5w         (H)   9) hwloc/2.11.3-mpich-level-zero      15) pti-gpu/0.11.0
          4) mpc/1.3.1-rdrlvsl          (H)  10) yaksa/0.3-7ks5f26             (H)  16) frameworks/2025.0.0
          5) gcc/13.3.0                      11) mpich/opt/develop-git.6037a7a
          6) oneapi/release/2025.0.5         12) libfabric/1.22.0

          Where:
          H:  Hidden Module

      [2025-05-05-072647][I]   - Setting up venv from conda=/opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0...
      [2025-05-05-072647][I]   - Found conda at /opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0
      [2025-05-05-072647][I]   - No VIRTUAL_ENV found in environment!
      [2025-05-05-072647][I]   - Looking for venv in VENV_DIR=./venvs/aurora_nre_models_frameworks-2025.0.0...
      [2025-05-05-072647][I]   - Activating existing venv in VENV_DIR=venvs/aurora_nre_models_frameworks-2025.0.0
      [2025-05-05-072647][I]   - Using python from: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
      [2025-05-05-072647][I] [JOB]
      [2025-05-05-072647][I]   - Setting up job for foremans
      [2025-05-05-072647][I]   - Machine: aurora
      [2025-05-05-072647][I]   - Hostname: x4318c6s6b0n0
      [2025-05-05-072647][I] [ezpz_get_pbs_env]
      [2025-05-05-072647][I]   - hostfile=/var/spool/pbs/aux/4671985.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
      [2025-05-05-072647][I]   - jobenv_file=/home/foremans/.pbsenv
      [2025-05-05-072648][I] [HOSTS]
      [2025-05-05-072648][I]   - HOSTFILE=/var/spool/pbs/aux/4671985.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
      [2025-05-05-072648][I]   - NHOSTS=2
      [2025-05-05-072648][I]   - HOSTS:
      [2025-05-05-072648][I]     - [host:0] - x4318c6s5b0n0.hostmgmt2318.cm.aurora.alcf.anl.gov
      [2025-05-05-072648][I]     - [host:1] - x4318c6s6b0n0.hostmgmt2318.cm.aurora.alcf.anl.gov
      [2025-05-05-072648][I] [DIST_INFO]
      [2025-05-05-072648][I]   - HOSTFILE=/var/spool/pbs/aux/4671985.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
      [2025-05-05-072648][I]   - NHOSTS=2
      [2025-05-05-072648][I]   - NGPU_PER_HOST=12
      [2025-05-05-072648][I]   - NGPUS=24
      [2025-05-05-072648][I] [LAUNCH]
      [2025-05-05-072648][I]   - To launch across all available GPUs, use: 'launch'
      [2025-05-05-072648][I]     launch = mpiexec --verbose --envall -n 24 -ppn 12 --hostfile /var/spool/pbs/aux/4671985.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind depth -d 8 --no-vni
      [2025-05-05-072648][I]   - Run 'which launch' to ensure that the alias is set correctly
      [2025-05-05-072648][I] ===== Environment Setup Complete =====
      took: 0h:00m:03s
      ```

    </details>

 

## 🔍 Environment Setup with `ezpz_setup_env`

- Wrapper around `ezpz_setup_job` `&&` `ezpz_setup_python`

1.  `ezpz_setup_job`: Determine the specifics of our active (PBS, SLURM)
    job[^3]

2.  `ezpz_setup_python`:

    - **if @ ALCF**:
      - Load the appropriate modules and activate base `conda` env
    - **else**:
      - Look for an active `conda` environment
        - If found, use it to build a new virtual environment
    - Activate the newly created `venvs/$(basename ${CONDA_PREFIX})`
      environment

## ⏱️ Working with Job Scheduler(s)

- `ezpz` integrates directly with the ALCF job scheduler[^4]
  - has mechanisms for getting information about our currently running
    jobs
- 🪄 *Automagically*:
  - Determine the specifics of our active (PBS, SLURM) job  
    (e.g. `${NHOSTS}`, `${NGPU_PER_HOST}`, `${NGPUS}`, …)
  - Load the appropriate modules[^5]
  - Create (or activate) a virtual environment *on top* of a base conda
    environment

## 🔄 Use Custom Node Lists

- Experiment[^6] with custom `hostfile`(s), e.g.:

  ``` bash
  source <(curl -L https://bit.ly/ezpz-utils)
  # 1. If no `hostfile` specified, find and use `$PBS_NODEFILE` 
  ezpz_setup_job
  # 2. Grab a subset of nodes:
  head -n 2 $PBS_NODEFILE > nodefile-0-1
  # 3. Pass custom `nodefile-0-1`:
  ezpz_setup_job nodefile-0-1  # will use `nodefile-0-1`
  ```

## 🐍 Python Environments

- **ALWAYS** work inside a virtual environment
  - best practice is to maintain separate virtual environments for:
    - each project you work on
    - different versions of a specific package you’re working with  
      e.g you would want different envs for `torch==2.X` vs `torch==2.Y`
  - *Mangled python environments are one of the most common issues faced
    by users*

## 📦 Install `ezpz`

1.  Install[^7]:

    ``` bash
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

2.  Run distributed test:

    ``` bash
    ezpz-test
    ```

3.  Launch *any* python *from* python

    - Launch a module:

      ``` bash
      ezpz-launch -m ezpz.test_dist
      ```

    - Launch a python string:

      ``` bash
      ezpz-launch -c "'import ezpz; ezpz.setup_torch()'"
      ```

## ➕ How to Modify Existing Code

``` diff
+ import ezpz
+ _ = ezpz.setup_torch()

- model.to('cuda')
+ model.to(ezpz.get_torch_device_type())
```

## ✨ Features

- Initializing PyTorch across multiple processes

  ``` python
  import ezpz
  _ = ezpz.setup_torch()
  rank = ezpz.get_rank()
  world_size = ezpz.get_world_size()
  local_rank = ezpz.get_local_rank()
  ```

- Automatic device detection (`xpu`, `cuda`, `mps`, `cpu`, …)

  ``` python
  x = torch.rand((10, 10)).to(ezpz.get_torch_device_type())
  ```

- Automatic (single-process) logging

  ``` python
  logger = ezpz.get_logger(__name__)
  ```

- Distributed debugger:

  ``` python
  try:
      buggy_code()
  except Exception:
      ezpz.breakpoint(0)
  ```

## 🧪 Experiment Tracking

<div class="block-code">

``` python
import ezpz
rank = ezpz.setup_torch()
logger = ezpz.get_logger(__name__)
if rank == 0:                   # -- [1.] --
    try:
        _ = ezpz.setup_wandb(
            "ezpz.examples.minimal"
        )
    except Exception:
        logger.exception(
            "Failed to initialize wandb, continuing without it"
        )

# ...build {model, optimizer}, etc...

for i in range(train_iters):
    metrics = train_step(...)
    logger.info(                 # -- [2.] --
        history.update(metrics)  # -- [3.] --
    )

if rank == 0:
    history.finalize()
```

1.  Initialize W&B (if `WANDB_DISABLED` is not set)
2.  Log summary of metrics to stdout
3.  Update `history.history` with metrics[^8]

</div>

## 🤏 Minimal Example

- See
  [`ezpz/examples/minimal.py`](https://github.com/saforem2/ezpz/blob/main/src/ezpz/examples/minimal.py)

<div class="block-code">

``` python
import os
import time
import ezpz
import torch

logger = ezpz.get_logger(__name__)


class Network(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sizes: list[int] | None,
    ):
        super(Network, self).__init__()
        nh = output_dim if sizes is None else sizes[0]
        layers = [torch.nn.Linear(input_dim, nh), torch.nn.ReLU()]
        if sizes is not None and len(sizes) > 1:
            for idx, size in enumerate(sizes[1:]):
                layers.extend(
                    [torch.nn.Linear(sizes[idx], size), torch.nn.ReLU()]
                )
            layers.append(torch.nn.Linear(sizes[-1], output_dim))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


@ezpz.timeitlogit(rank=ezpz.get_rank())
def train(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> ezpz.History:
    unwrapped_model = (
        model.module
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model
    )
    history = ezpz.History()
    device_type = ezpz.get_torch_device_type()
    dtype = unwrapped_model.layers[0].weight.dtype
    bsize = int(os.environ.get("BATCH_SIZE", 64))
    isize = unwrapped_model.layers[0].in_features
    warmup = int(os.environ.get("WARMUP_ITERS", 10))
    log_freq = int(os.environ.get("LOG_FREQ", 1))
    model.train()
    for step in range(int(os.environ.get("TRAIN_ITERS", 500))):
        with torch.autocast(
            device_type=device_type,
            dtype=dtype,
        ):
            t0 = time.perf_counter()
            x = torch.rand((bsize, isize), dtype=dtype).to(device_type)
            y = model(x)
            loss = ((y - x) ** 2).sum()
            dtf = (t1 := time.perf_counter()) - t0
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            dtb = time.perf_counter() - t1
            if step % log_freq == 0 and step > warmup:
                logger.info(
                    history.update(
                        {
                            "iter": step,
                            "loss": loss.item(),
                            "dt": dtf + dtb,
                            "dtf": dtf,
                            "dtb": dtb,
                        }
                    )
                )
    return history


@ezpz.timeitlogit(rank=ezpz.get_rank())
def setup():
    rank = ezpz.setup_torch()
    if os.environ.get("WANDB_DISABLED", False):
        logger.info("WANDB_DISABLED is set, not initializing wandb")
    elif rank == 0:
        try:
            _ = ezpz.setup_wandb(
                project_name=os.environ.get(
                    "PROJECT_NAME", "ezpz.examples.minimal"
                )
            )
        except Exception:
            logger.exception(
                "Failed to initialize wandb, continuing without it"
            )
    device_type = ezpz.get_torch_device_type()
    model = Network(
        input_dim=int((os.environ.get("INPUT_SIZE", 128))),
        output_dim=int(os.environ.get("OUTPUT_SIZE", 128)),
        sizes=[
            int(x)
            for x in os.environ.get("LAYER_SIZES", "1024,512,256,128").split(
                ","
            )
        ],
    )
    model.to(device_type)
    model.to((os.environ.get("DTYPE", torch.bfloat16)))
    logger.info(f"{model=}")
    optimizer = torch.optim.Adam(model.parameters())
    if ezpz.get_world_size() > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(model, device_ids=[ezpz.get_local_rank()])

    return model, optimizer


def main():
    model, optimizer = setup()
    history = train(model, optimizer)
    if ezpz.get_rank() == 0:
        dataset = history.finalize()
        logger.info(f"{dataset=}")


if __name__ == "__main__":
    main()
```

</div>

## 🏃‍♂️ Running the Minimal Example

To run the previous example we:

1.  Source the `ezpz` utils script:

    ``` bash
    source <(curl -L https://bit.ly/ezpz-utils)
    ```

2.  Setup our environment:

    ``` bash
    ezpz_setup_env
    ```

3.  Run the example:

    ``` bash
    ezpz-launch -m ezpz.examples.minimal
    ```

<!-- -->

1.  <details closed>

    <summary>

    Output:
    </summary>

    ``` bash
    #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
    #[/f/d/f/p/s/ezpz][🌱 update-utils][📦📝🤷✓] [⏱️ 5m23s]
    #[05/06/25 @ 09:06:04][x4000c2s6b0n0]
    ; ezpz-launch -m ezpz.examples.minimal
    [W506 09:06:14.877537382 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
    Overriding a previously registered kernel for the same operator and the same dispatch key
    operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
        registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
    dispatch key: XPU
    previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
        new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
    [2025-05-06 09:06:18,965] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to xpu (auto detect)
    [2025-05-06 09:06:21][I][ezpz/launch:157] Job ID: 4673761
    [2025-05-06 09:06:21][I][ezpz/launch:163] Node file: /var/spool/pbs/aux/4673761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
    [2025-05-06 09:06:21][I][ezpz/launch:178] Building command to execute by piecing together:(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']
    [2025-05-06 09:06:21][I][ezpz/launch:182] (1.) ['launch_cmd']: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4673761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8
    [2025-05-06 09:06:21][I][ezpz/launch:183] (2.) ['python']: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
    [2025-05-06 09:06:21][I][ezpz/launch:184] (3.) ['cmd_to_launch']:  -m ezpz.examples.minimal
    [2025-05-06 09:06:21][I][ezpz/launch:189] Took: 0.43 seconds to build command.
    [2025-05-06 09:06:21][I][ezpz/launch:192] Executing: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4673761.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -m ezpz.examples.minimal
    [2025-05-06 09:06:21][I][ezpz/launch:119] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
    [2025-05-06 09:06:21][I][ezpz/launch:199] Execution started @ 2025-05-06-090621...

    Disabling local launch: multi-node application
    Connected to tcp://x4000c2s6b0n0.hostmgmt2000.cm.aurora.alcf.anl.gov:7919
    Launching application 9237e362-f53a-4401-8cab-78cc0b54ab87
    [2025-05-06 09:06:45][I][ezpz/dist:567] Using get_torch_device_type()='xpu' with backend='ccl'
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 4/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 8/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 9/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][10/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][11/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 5/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 1/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 2/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 3/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 6/23]
    [2025-05-06 09:06:45][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 7/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][12/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][13/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][16/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][17/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][14/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][15/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][21/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][20/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][23/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][22/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][19/23]
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s7b0n0'][18/23]
    [2025-05-06 09:06:46][I][ezpz/dist:947] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
    [2025-05-06 09:06:46][I][ezpz/dist:994] ['x4000c2s6b0n0'][ 0/23]
    2025:05:06-09:06:46:(19763) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
    [2025-05-06 09:06:47][I][ezpz/dist:1217] Setting up wandb from rank=0
    [2025-05-06 09:06:47][I][ezpz/dist:1218] Using WB_PROJECT=ezpz.examples.minimal
    wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
    wandb: Tracking run with wandb version 0.19.10
    wandb: Run data is saved locally in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250506_090647-q9u196rq
    wandb: Run `wandb offline` to turn off syncing.
    wandb: Syncing run pretty-paper-29
    wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz.examples.minimal
    wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz.examples.minimal/runs/q9u196rq
    [2025-05-06 09:06:47][I][ezpz/dist:1246] wandb.run=[pretty-paper-29](https://wandb.ai/aurora_gpt/ezpz.examples.minimal/runs/q9u196rq)
    [2025-05-06 09:06:47][I][ezpz/dist:1286] Running on machine='Aurora'
    [2025-05-06 09:06:47][I][examples/minimal:104:__main__] model=Network(
    (layers): Sequential(
        (0): Linear(in_features=128, out_features=1024, bias=True)
        (1): ReLU()
        (2): Linear(in_features=1024, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=256, bias=True)
        (5): ReLU()
        (6): Linear(in_features=256, out_features=128, bias=True)
        (7): ReLU()
        (8): Linear(in_features=128, out_features=128, bias=True)
    )
    )
    [2025-05-06 09:06:58][I][ezpz/dist:143] `setup` took: dt=13.7828s
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=0 loss=2701.321045 dt=0.623345 dtf=0.381410 dtb=0.241935
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=1 loss=2527.130371 dt=0.151625 dtf=0.002179 dtb=0.149447
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=2 loss=2318.325195 dt=0.003961 dtf=0.000944 dtb=0.003016
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=3 loss=1952.584473 dt=0.003688 dtf=0.000970 dtb=0.002718
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=4 loss=1793.388062 dt=0.003742 dtf=0.001064 dtb=0.002677
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=5 loss=1555.838867 dt=0.003606 dtf=0.000944 dtb=0.002662
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=6 loss=1234.822510 dt=0.003723 dtf=0.000970 dtb=0.002753
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=7 loss=1117.542969 dt=0.003695 dtf=0.000956 dtb=0.002739
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=8 loss=1010.627075 dt=0.003899 dtf=0.000984 dtb=0.002915
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=9 loss=907.192017 dt=0.003738 dtf=0.000963 dtb=0.002775
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=10 loss=911.176147 dt=0.003876 dtf=0.000940 dtb=0.002936
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=11 loss=826.104065 dt=0.003670 dtf=0.000904 dtb=0.002766
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=12 loss=768.030396 dt=0.003839 dtf=0.000900 dtb=0.002940
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=13 loss=754.958557 dt=0.003710 dtf=0.000906 dtb=0.002804
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=14 loss=750.200745 dt=0.003722 dtf=0.000885 dtb=0.002837
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=15 loss=727.392395 dt=0.003824 dtf=0.000897 dtb=0.002928
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=16 loss=721.139099 dt=0.003677 dtf=0.000923 dtb=0.002754
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=17 loss=715.588501 dt=0.003681 dtf=0.000923 dtb=0.002758
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=18 loss=711.832520 dt=0.004013 dtf=0.000902 dtb=0.003110
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=19 loss=712.932617 dt=0.003716 dtf=0.000922 dtb=0.002794
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=20 loss=702.517212 dt=0.003796 dtf=0.000895 dtb=0.002901
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=21 loss=698.924438 dt=0.003716 dtf=0.000901 dtb=0.002815
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=22 loss=697.166931 dt=0.003972 dtf=0.001139 dtb=0.002832
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=23 loss=706.649780 dt=0.003700 dtf=0.000909 dtb=0.002791
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=24 loss=703.272400 dt=0.003783 dtf=0.000901 dtb=0.002882
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=25 loss=709.477356 dt=0.003557 dtf=0.000896 dtb=0.002661
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=26 loss=722.453125 dt=0.003578 dtf=0.000899 dtb=0.002679
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=27 loss=708.771179 dt=0.003554 dtf=0.000886 dtb=0.002668
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=28 loss=702.787598 dt=0.003620 dtf=0.000922 dtb=0.002698
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=29 loss=688.691895 dt=0.003543 dtf=0.000890 dtb=0.002653
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=30 loss=677.675781 dt=0.003570 dtf=0.000887 dtb=0.002683
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=31 loss=705.331299 dt=0.003538 dtf=0.000896 dtb=0.002643
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=32 loss=686.603394 dt=0.003586 dtf=0.000915 dtb=0.002671
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=33 loss=686.867798 dt=0.003723 dtf=0.000902 dtb=0.002821
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=34 loss=691.201904 dt=0.004015 dtf=0.000893 dtb=0.003122
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=35 loss=689.949707 dt=0.003646 dtf=0.000904 dtb=0.002741
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=36 loss=668.631348 dt=0.003907 dtf=0.000918 dtb=0.002989
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=37 loss=684.760254 dt=0.003613 dtf=0.000895 dtb=0.002718
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=38 loss=666.486328 dt=0.003729 dtf=0.000903 dtb=0.002826
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=39 loss=680.438721 dt=0.003700 dtf=0.000890 dtb=0.002810
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=40 loss=668.775513 dt=0.003776 dtf=0.000916 dtb=0.002860
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=41 loss=673.034912 dt=0.003967 dtf=0.000952 dtb=0.003015
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=42 loss=674.066772 dt=0.003890 dtf=0.000963 dtb=0.002927
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=43 loss=673.859985 dt=0.003640 dtf=0.000909 dtb=0.002730
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=44 loss=667.940552 dt=0.003580 dtf=0.000901 dtb=0.002679
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=45 loss=678.843750 dt=0.003621 dtf=0.000913 dtb=0.002708
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=46 loss=687.354187 dt=0.003796 dtf=0.000898 dtb=0.002898
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=47 loss=685.980774 dt=0.003620 dtf=0.000911 dtb=0.002708
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=48 loss=669.822632 dt=0.003582 dtf=0.000905 dtb=0.002677
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=49 loss=681.426880 dt=0.003730 dtf=0.000945 dtb=0.002785
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=50 loss=682.930542 dt=0.003701 dtf=0.000946 dtb=0.002756
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=51 loss=676.441895 dt=0.003657 dtf=0.000931 dtb=0.002726
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=52 loss=664.631531 dt=0.003676 dtf=0.000946 dtb=0.002730
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=53 loss=669.697571 dt=0.003805 dtf=0.000913 dtb=0.002892
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=54 loss=665.016602 dt=0.003814 dtf=0.000946 dtb=0.002867
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=55 loss=672.755981 dt=0.003617 dtf=0.000912 dtb=0.002705
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=56 loss=676.824341 dt=0.003804 dtf=0.000924 dtb=0.002880
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=57 loss=676.435181 dt=0.003807 dtf=0.000937 dtb=0.002870
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=58 loss=680.153992 dt=0.003991 dtf=0.000937 dtb=0.003054
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=59 loss=675.248108 dt=0.003597 dtf=0.000892 dtb=0.002705
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=60 loss=673.595093 dt=0.003694 dtf=0.000911 dtb=0.002783
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=61 loss=686.233032 dt=0.003583 dtf=0.000900 dtb=0.002683
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=62 loss=682.671265 dt=0.003702 dtf=0.000908 dtb=0.002793
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=63 loss=673.332092 dt=0.003626 dtf=0.000896 dtb=0.002731
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=64 loss=678.947998 dt=0.003721 dtf=0.000903 dtb=0.002818
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=65 loss=664.849792 dt=0.003625 dtf=0.000912 dtb=0.002713
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=66 loss=671.088013 dt=0.003731 dtf=0.000893 dtb=0.002837
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=67 loss=676.324768 dt=0.003726 dtf=0.000937 dtb=0.002789
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=68 loss=664.155518 dt=0.003764 dtf=0.000973 dtb=0.002791
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=69 loss=674.292114 dt=0.003703 dtf=0.000935 dtb=0.002769
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=70 loss=668.928772 dt=0.003908 dtf=0.000936 dtb=0.002972
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=71 loss=675.064697 dt=0.003670 dtf=0.000921 dtb=0.002748
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=72 loss=677.371338 dt=0.003632 dtf=0.000964 dtb=0.002667
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=73 loss=685.282959 dt=0.003582 dtf=0.000894 dtb=0.002688
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=74 loss=669.304443 dt=0.003767 dtf=0.000908 dtb=0.002859
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=75 loss=676.679932 dt=0.003779 dtf=0.000904 dtb=0.002875
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=76 loss=678.548462 dt=0.004022 dtf=0.000921 dtb=0.003101
    [2025-05-06 09:06:59][I][examples/minimal:61:__main__] iter=77 loss=673.683105 dt=0.003715 dtf=0.000910 dtb=0.002805
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=78 loss=676.570129 dt=0.003722 dtf=0.000921 dtb=0.002801
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=79 loss=681.414795 dt=0.003569 dtf=0.000907 dtb=0.002662
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=80 loss=680.041992 dt=0.003691 dtf=0.000918 dtb=0.002773
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=81 loss=675.775024 dt=0.003611 dtf=0.000897 dtb=0.002714
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=82 loss=670.443359 dt=0.003796 dtf=0.000910 dtb=0.002886
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=83 loss=660.718018 dt=0.003568 dtf=0.000900 dtb=0.002669
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=84 loss=672.146912 dt=0.003607 dtf=0.000923 dtb=0.002684
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=85 loss=676.868896 dt=0.003542 dtf=0.000918 dtb=0.002624
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=86 loss=678.217529 dt=0.003735 dtf=0.000898 dtb=0.002838
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=87 loss=665.618103 dt=0.003579 dtf=0.000909 dtb=0.002670
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=88 loss=668.519287 dt=0.003574 dtf=0.000903 dtb=0.002671
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=89 loss=664.486694 dt=0.003928 dtf=0.000942 dtb=0.002985
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=90 loss=677.690918 dt=0.003746 dtf=0.000966 dtb=0.002780
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=91 loss=668.240601 dt=0.003564 dtf=0.000894 dtb=0.002670
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=92 loss=660.485474 dt=0.003608 dtf=0.000909 dtb=0.002700
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=93 loss=664.691772 dt=0.003570 dtf=0.000913 dtb=0.002657
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=94 loss=656.607910 dt=0.003601 dtf=0.000910 dtb=0.002691
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=95 loss=670.816650 dt=0.003555 dtf=0.000904 dtb=0.002652
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=96 loss=663.897339 dt=0.003560 dtf=0.000895 dtb=0.002665
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=97 loss=659.260620 dt=0.003908 dtf=0.000941 dtb=0.002967
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=98 loss=660.536499 dt=0.003615 dtf=0.000897 dtb=0.002718
    [2025-05-06 09:07:00][I][examples/minimal:61:__main__] iter=99 loss=661.475586 dt=0.003756 dtf=0.000946 dtb=0.002809
    [2025-05-06 09:07:00][I][ezpz/dist:143] `train`((DistributedDataParallel(
    (module): Network(
        (layers): Sequential(
        (0): Linear(in_features=128, out_features=1024, bias=True)
        (1): ReLU()
        (2): Linear(in_features=1024, out_features=512, bias=True)
        (3): ReLU()
        (4): Linear(in_features=512, out_features=256, bias=True)
        (5): ReLU()
        (6): Linear(in_features=256, out_features=128, bias=True)
        (7): ReLU()
        (8): Linear(in_features=128, out_features=128, bias=True)
        )
    )
    ), Adam (
    Parameter Group 0
        amsgrad: False
        betas: (0.9, 0.999)
        capturable: False
        differentiable: False
        eps: 1e-08
        foreach: None
        fused: None
        lr: 0.001
        maximize: False
        weight_decay: 0
    ))) took: dt=1.2669s
    [2025-05-06 09:07:02][I][ezpz/history:721] Saving iter plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/mplot
    [2025-05-06 09:07:02][I][ezpz/history:721] Saving loss plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/mplot
    [2025-05-06 09:07:02][I][ezpz/history:721] Saving dt plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/mplot
    [2025-05-06 09:07:02][I][ezpz/history:721] Saving dtf plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/mplot
    [2025-05-06 09:07:03][I][ezpz/history:721] Saving dtb plot to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/mplot
    [2025-05-06 09:07:03][I][ezpz/history:618] Saving tplots to /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot
                        loss [2025-05-06-090703]
          ┌────────────────────────────────────────────────────┐
    2701.3┤▌                                                   │
          │▐                                                   │
    2360.5┤▝▖                                                  │
          │ ▌                                                  │
          │ ▌                                                  │
    2019.8┤ ▚                                                  │
          │ ▝▖                                                 │
    1679.0┤  ▌                                                 │
          │  ▐                                                 │
    1338.2┤  ▐                                                 │
          │  ▝▖                                                │
          │   ▐                                                │
     997.4┤    ▚▖                                              │
          │     ▚▖                                             │
     656.6┤      ▝▀▀▀▀▀▀▀▀▄▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
          └─┬─┬──┬──┬───┬──┬──┬────┬────┬──┬───┬──┬───┬──┬───┬─┘
          0 3 7 13 19  27 33 37   47   57 64  70 77  84 91  98
    loss                           iter
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/loss.txt
                        dt [2025-05-06-090703]
        ┌──────────────────────────────────────────────────────┐
    0.62┤▌                                                     │
        │▌                                                     │
    0.52┤▌                                                     │
        │▌                                                     │
        │▌                                                     │
    0.42┤▌                                                     │
        │▌                                                     │
    0.31┤▌                                                     │
        │▌                                                     │
    0.21┤▌                                                     │
        │▌                                                     │
        │▐                                                     │
    0.11┤▐                                                     │
        │▐                                                     │
    0.00┤▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
        └─┬─┬──┬───┬───┬──┬────┬──┬────┬──┬───┬───┬──┬───┬───┬─┘
        0 3 7 13  19  27 33   42 47   57 62  70  77 84  91  98
    dt                            iter
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dt.txt
                        dt [2025-05-06-090703]
        ┌──────────────────────────────────────────────────────┐
    98.0┤█████                                                 │
        │█████                                                 │
    81.7┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    65.3┤█████                                                 │
        │█████                                                 │
    49.0┤█████                                                 │
        │█████                                                 │
    32.7┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    16.3┤█████                                                 │
        │█████                                                 │
     0.0┤█████      █████                                 █████│
        └┬────────────┬─────────────┬────────────┬────────────┬┘
    -0.02        0.14          0.31         0.48        0.65
    freq                           dt
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dt-hist.txt
                        dtf [2025-05-06-090703]
         ┌─────────────────────────────────────────────────────┐
    0.381┤▌                                                    │
         │▌                                                    │
    0.318┤▌                                                    │
         │▌                                                    │
         │▌                                                    │
    0.255┤▌                                                    │
         │▌                                                    │
    0.191┤▌                                                    │
         │▌                                                    │
    0.128┤▌                                                    │
         │▌                                                    │
         │▌                                                    │
    0.064┤▌                                                    │
         │▌                                                    │
    0.001┤▚▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
         └─┬─┬──┬──┬────┬──┬──┬───┬────┬──┬───┬───┬───┬──┬───┬─┘
        0 3 7 13 19   27 33 39  47   57 62  70  77  84 91  98
    dtf                           iter
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dtf.txt
                        dtf [2025-05-06-090703]
        ┌──────────────────────────────────────────────────────┐
    99.0┤█████                                                 │
        │█████                                                 │
    82.5┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    66.0┤█████                                                 │
        │█████                                                 │
    49.5┤█████                                                 │
        │█████                                                 │
    33.0┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    16.5┤█████                                                 │
        │█████                                                 │
     0.0┤█████                                            █████│
        └┬────────────┬─────────────┬────────────┬────────────┬┘
     -0.02        0.09          0.19         0.29        0.40
    freq                           dtf
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dtf-hist.txt
                        dtb [2025-05-06-090703]
         ┌─────────────────────────────────────────────────────┐
    0.242┤▌                                                    │
         │▌                                                    │
    0.202┤▌                                                    │
         │▌                                                    │
         │▌                                                    │
    0.162┤▚                                                    │
         │▐                                                    │
    0.122┤▐                                                    │
         │▐                                                    │
    0.082┤▐                                                    │
         │▐                                                    │
         │▐                                                    │
    0.043┤▐                                                    │
         │▐                                                    │
    0.003┤▝▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
         └─┬─┬──┬──┬────┬──┬──┬───┬────┬──┬───┬───┬───┬──┬───┬─┘
         0 3 7 13 19   27 33 39  47   57 62  70  77  84 91  98
    dtb                           iter
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dtb.txt
                        dtb [2025-05-06-090703]
        ┌──────────────────────────────────────────────────────┐
    98.0┤█████                                                 │
        │█████                                                 │
    81.7┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    65.3┤█████                                                 │
        │█████                                                 │
    49.0┤█████                                                 │
        │█████                                                 │
    32.7┤█████                                                 │
        │█████                                                 │
        │█████                                                 │
    16.3┤█████                                                 │
        │█████                                                 │
     0.0┤█████                           ██████           █████│
        └┬────────────┬─────────────┬────────────┬────────────┬┘
    -0.008        0.057         0.122        0.187      0.253
    freq                           dtb
    text saved in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/plots/tplot/dtb-hist.txt
    [2025-05-06 09:07:03][I][ezpz/utils:198] Saving dataset to: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/outputs/History-2025-05-06-090700/2025-05-06-090700/History-2025-05-06-090700/dataset_dataset.h5
    wandb:
    wandb: 🚀 View run pretty-paper-29 at: https://wandb.ai/aurora_gpt/ezpz.examples.minimal/runs/q9u196rq
    wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250506_090647-q9u196rq/logs
    Application 9237e362 resources: utime=843s stime=176s maxrss=4006656KB inblock=668002 oublock=1640 minflt=11466255 majflt=45004 nvcsw=498142 nivcsw=5295709
    [2025-05-06 09:07:06][I][ezpz/launch:201] Execution finished @ 2025-05-06-090706
    [2025-05-06 09:07:06][I][ezpz/launch:202] Command took 44.95 seconds to run. Exiting.
    took: 0h:00m:56s
    ```

    </details>

## 📝 `ezpz-test`

- `ezpz-test` is a simple test script that trains a small model using
  DDP across all available GPUs

  - It will automatically detect the number of GPUs and launch an
    appropriate `mpiexec` command to run the training script across all
    GPUs

- See:
  [ezpz/test.py](https://github.com/ezpz/ezpz/blob/main/ezpz/test.py)

- Command:

  ``` bash
  #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
  #[05/05/25 @ 07:41:35][x4520c1s0b0n0][/f/d/f/p/s/ezpz][🌱 update-utils][📦🤷✓] [⏱️ 54s]
  ; ezpz-test
  ```

## 🦜 Generate Text

- See:
  [ezpz/generate.py](https://github.com/ezpz/ezpz/blob/main/ezpz/generate.py)

- Command:

  ``` bash
  python3 -m ezpz.generate --model_name meta-llama/Llama-3.1-8B
  ```

## 🤗 Huggingface Trainer

- See
  [ezpz/hf_trainer.py](https://github.com/ezpz/ezpz/blob/main/ezpz/hf_trainer.py)

- Command:

  ``` bash
  ezpz-launch -m ezpz.hf_trainer \
      --dataset_name=eliplutchok/fineweb-small-sample \
      --streaming \
      --model_name_or_path=meta-llama/Llama-3.2-1B \
      --bf16=true \
      --do_train=true \
      --do_eval=true \
      --report-to=wandb \
      --logging-steps=1 \
      --include-tokens-per-second=true \
      --block-size=128 \
      --max-steps=10 \
      --include-num-input-tokens-seen=true \
      --auto_find_batch_size=true \
      --gradient_checkpointing=true \
      --optim=adamw_torch \
      --overwrite-output-dir=true \
      --logging-first-step \
      --include-for-metrics='inputs,loss' \
      --max-eval-samples=50 \
      --ddp-backend=ccl
  ```

<!-- -->

- <details closed>

  <summary>

  Output:
  </summary>

  ``` bash

  #[🐍 aurora_nre_models_frameworks-2025.0.0](👻 aurora_nre_models_frameworks-2025.0.0)
  #[/f/d/f/p/s/ezpz][🌱 update-utils][📦📝🤷✓] [⏱️ 1m54s]
  #[05/06/25 @ 22:25:54][x4505c5s7b0n0]
  ; ezpz-launch -m ezpz.hf_trainer --dataset_name=eliplutchok/fineweb-small-sample --streaming --model_name_or_path=meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --block-size=128 --max-steps=10 --include-num-input-tokens-seen=true --auto_find_batch_size=true --gradient_checkpointing=true --optim=adamw_torch --overwrite-output-dir=true --logging-first-step --include-for-metrics='inputs,loss' --max-eval-samples=50 --ddp-backend=ccl # --fsdp=shard_grad_op
  [W506 22:25:56.901078167 OperatorEntry.cpp:155] Warning: Warning only once for all operators,  other operators may also be overridden.
  Overriding a previously registered kernel for the same operator and the same dispatch key
  operator: aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
      registered at /build/pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
  dispatch key: XPU
  previous kernel: registered at /build/pytorch/build/aten/src/ATen/RegisterCPU.cpp:30476
      new kernel: registered at /build/intel-pytorch-extension/build/Release/csrc/gpu/csrc/aten/generated/ATen/RegisterXPU.cpp:2971 (function operator())
  [2025-05-06 22:26:00,816] [INFO] [real_accelerator.py:239:get_accelerator] Setting ds_accelerator to xpu (auto detect)
  [2025-05-06 22:26:02][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
  [2025-05-06 22:26:02][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
  [2025-05-06 22:26:03][I][ezpz/launch:157] Job ID: 4675836
  [2025-05-06 22:26:03][I][ezpz/launch:163] Node file: /var/spool/pbs/aux/4675836.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov
  [2025-05-06 22:26:03][I][ezpz/launch:178] Building command to execute by piecing together:(1.) ['launch_cmd'] + (2.) ['python'] + (3.) ['cmd_to_launch']
  [2025-05-06 22:26:03][I][ezpz/launch:182] (1.) ['launch_cmd']: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4675836.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8
  [2025-05-06 22:26:03][I][ezpz/launch:183] (2.) ['python']: /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3
  [2025-05-06 22:26:03][I][ezpz/launch:184] (3.) ['cmd_to_launch']:  -m ezpz.hf_trainer --dataset_name=eliplutchok/fineweb-small-sample --streaming --model_name_or_path=meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --block-size=128 --max-steps=10 --include-num-input-tokens-seen=true --auto_find_batch_size=true --gradient_checkpointing=true --optim=adamw_torch --overwrite-output-dir=true --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --ddp-backend=ccl
  [2025-05-06 22:26:03][I][ezpz/launch:189] Took: 0.45 seconds to build command.
  [2025-05-06 22:26:03][I][ezpz/launch:192] Executing: mpiexec --verbose --envall --np=24 --ppn=12 --hostfile=/var/spool/pbs/aux/4675836.aurora-pbs-0001.hostmgmt.cm.aurora.alcf.anl.gov --cpu-bind=depth --depth=8 /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/venvs/aurora_nre_models_frameworks-2025.0.0/bin/python3 -m ezpz.hf_trainer --dataset_name=eliplutchok/fineweb-small-sample --streaming --model_name_or_path=meta-llama/Llama-3.2-1B --bf16=true --do_train=true --do_eval=true --report-to=wandb --logging-steps=1 --include-tokens-per-second=true --block-size=128 --max-steps=10 --include-num-input-tokens-seen=true --auto_find_batch_size=true --gradient_checkpointing=true --optim=adamw_torch --overwrite-output-dir=true --logging-first-step --include-for-metrics=inputs,loss --max-eval-samples=50 --ddp-backend=ccl
  [2025-05-06 22:26:03][I][ezpz/launch:119] Filtering for Aurora-specific messages. To view list of filters, run with `EZPZ_LOG_LEVEL=DEBUG`
  [2025-05-06 22:26:03][I][ezpz/launch:199] Execution started @ 2025-05-06-222603...

  Disabling local launch: multi-node application
  Connected to tcp://x4505c5s6b0n0.hostmgmt2505.cm.aurora.alcf.anl.gov:7919
  Launching application 3917764c-4dd9-4d75-bed1-dd671fc83cba
  [2025-05-06 22:26:18][I][ezpz/__init__:278:ezpz] Setting logging level to 'INFO' on 'RANK == 0'
  [2025-05-06 22:26:18][I][ezpz/__init__:279:ezpz] Setting logging level to 'CRITICAL' on all others 'RANK != 0'
  [2025-05-06 22:26:19][I][ezpz/dist:567] Using get_torch_device_type()='xpu' with backend='ccl'
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 4/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 5/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 6/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 7/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][12/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][13/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][15/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][16/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][18/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][19/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][22/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][20/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 3/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][11/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][14/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][23/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 1/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 8/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 2/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][17/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][10/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s7b0n0'][21/23]
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 9/23]
  [2025-05-06 22:26:19][I][ezpz/dist:947] Using device='xpu' with backend='ddp' + 'ccl' for distributed training.
  [2025-05-06 22:26:19][I][ezpz/dist:994] ['x4505c5s6b0n0'][ 0/23]
  2025:05:06-22:26:19:(191240) |CCL_WARN| value of CCL_LOG_LEVEL changed to be error (default:warn)
  [2025-05-06 22:26:20][I][ezpz/dist:1217] Setting up wandb from rank=0
  [2025-05-06 22:26:20][I][ezpz/dist:1218] Using WB_PROJECT=ezpz-hf_trainer-meta-llama-Llama-3.2-1B
  wandb: Currently logged in as: foremans (aurora_gpt) to https://api.wandb.ai. Use `wandb login --relogin` to force relogin
  wandb: Tracking run with wandb version 0.19.10
  wandb: Run data is saved locally in /lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250506_222620-6yl6uks0
  wandb: Run `wandb offline` to turn off syncing.
  wandb: Syncing run cosmic-meadow-38
  wandb: ⭐️ View project at https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B
  wandb: 🚀 View run at https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/6yl6uks0
  [2025-05-06 22:26:21][I][ezpz/dist:1246] wandb.run=[cosmic-meadow-38](https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/6yl6uks0)
  [2025-05-06 22:26:21][I][ezpz/dist:1286] Running on machine='Aurora'
  [2025-05-06 22:26:21][W][utils/_logger:68:__main__] Process rank: 0, device: xpu:0, n_gpu: 1, distributed training: True
  [2025-05-06 22:26:21][I][ezpz/hf_trainer:437:__main__] Training/evaluation parameters TrainingArguments(
      _n_gpu=1,
      accelerator_config={'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None, 'use_configured_state': False},
      adafactor=False,
      adam_beta1=0.9,
      adam_beta2=0.999,
      adam_epsilon=1e-08,
      auto_find_batch_size=True,
      average_tokens_across_devices=False,
      batch_eval_metrics=False,
      bf16=True,
      bf16_full_eval=False,
      data_seed=None,
      dataloader_drop_last=False,
      dataloader_num_workers=0,
      dataloader_persistent_workers=False,
      dataloader_pin_memory=True,
      dataloader_prefetch_factor=None,
      ddp_backend=ccl,
      ddp_broadcast_buffers=None,
      ddp_bucket_cap_mb=None,
      ddp_find_unused_parameters=None,
      ddp_timeout=1800,
      debug=[],
      deepspeed=None,
      disable_tqdm=True,
      do_eval=True,
      do_predict=False,
      do_train=True,
      eval_accumulation_steps=None,
      eval_delay=0,
      eval_do_concat_batches=True,
      eval_on_start=False,
      eval_steps=None,
      eval_strategy=no,
      eval_use_gather_object=False,
      fp16=False,
      fp16_backend=auto,
      fp16_full_eval=False,
      fp16_opt_level=O1,
      fsdp=[],
      fsdp_config={'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False},
      fsdp_min_num_params=0,
      fsdp_transformer_layer_cls_to_wrap=None,
      full_determinism=False,
      gradient_accumulation_steps=1,
      gradient_checkpointing=True,
      gradient_checkpointing_kwargs=None,
      greater_is_better=None,
      group_by_length=False,
      half_precision_backend=auto,
      hub_always_push=False,
      hub_model_id=None,
      hub_private_repo=None,
      hub_strategy=every_save,
      hub_token=<HUB_TOKEN>,
      ignore_data_skip=False,
      include_for_metrics=['inputs,loss'],
      include_inputs_for_metrics=False,
      include_num_input_tokens_seen=True,
      include_tokens_per_second=True,
      jit_mode_eval=False,
      label_names=None,
      label_smoothing_factor=0.0,
      learning_rate=5e-05,
      length_column_name=length,
      load_best_model_at_end=False,
      local_rank=0,
      log_level=passive,
      log_level_replica=warning,
      log_on_each_node=True,
      logging_dir=trainer_output/runs/May06_22-26-20_x4505c5s6b0n0,
      logging_first_step=True,
      logging_nan_inf_filter=True,
      logging_steps=1.0,
      logging_strategy=steps,
      lr_scheduler_kwargs={},
      lr_scheduler_type=linear,
      max_grad_norm=1.0,
      max_steps=10,
      metric_for_best_model=None,
      mp_parameters=,
      neftune_noise_alpha=None,
      num_train_epochs=3.0,
      optim=adamw_torch,
      optim_args=None,
      optim_target_modules=None,
      output_dir=trainer_output,
      overwrite_output_dir=True,
      past_index=-1,
      per_device_eval_batch_size=8,
      per_device_train_batch_size=8,
      prediction_loss_only=False,
      push_to_hub=False,
      push_to_hub_model_id=None,
      push_to_hub_organization=None,
      push_to_hub_token=<PUSH_TO_HUB_TOKEN>,
      ray_scope=last,
      remove_unused_columns=True,
      report_to=['wandb'],
      restore_callback_states_from_checkpoint=False,
      resume_from_checkpoint=None,
      run_name=trainer_output,
      save_on_each_node=False,
      save_only_model=False,
      save_safetensors=True,
      save_steps=500,
      save_strategy=steps,
      save_total_limit=None,
      seed=42,
      skip_memory_metrics=True,
      tf32=None,
      torch_compile=False,
      torch_compile_backend=None,
      torch_compile_mode=None,
      torch_empty_cache_steps=None,
      torchdynamo=None,
      tp_size=0,
      tpu_metrics_debug=False,
      tpu_num_cores=None,
      use_cpu=False,
      use_ipex=False,
      use_legacy_prediction_loop=False,
      use_liger_kernel=False,
      use_mps_device=False,
      warmup_ratio=0.0,
      warmup_steps=0,
      weight_decay=0.0,
  )
  [INFO|configuration_utils.py:693] 2025-05-06 22:26:24,266 >> loading configuration file config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/config.json
  [INFO|configuration_utils.py:765] 2025-05-06 22:26:24,267 >> Model config LlamaConfig {
  "architectures": [
      "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 2048,
  "initializer_range": 0.02,
  "intermediate_size": 8192,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 16,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
      "factor": 32.0,
      "high_freq_factor": 4.0,
      "low_freq_factor": 1.0,
      "original_max_position_embeddings": 8192,
      "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.3",
  "use_cache": true,
  "vocab_size": 128256
  }

  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file tokenizer.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/tokenizer.json
  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file tokenizer.model from cache at None
  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file added_tokens.json from cache at None
  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file special_tokens_map.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/special_tokens_map.json
  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file tokenizer_config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/tokenizer_config.json
  [INFO|tokenization_utils_base.py:2060] 2025-05-06 22:26:24,312 >> loading file chat_template.jinja from cache at None
  [INFO|tokenization_utils_base.py:2323] 2025-05-06 22:26:24,692 >> Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
  [INFO|modeling_utils.py:1124] 2025-05-06 22:26:24,704 >> loading weights file model.safetensors from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/model.safetensors
  [INFO|configuration_utils.py:1142] 2025-05-06 22:26:24,708 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "eos_token_id": 128001
  }

  [INFO|modeling_utils.py:4930] 2025-05-06 22:26:32,810 >> All model checkpoint weights were used when initializing LlamaForCausalLM.

  [INFO|modeling_utils.py:4938] 2025-05-06 22:26:32,810 >> All the weights of LlamaForCausalLM were initialized from the model checkpoint at meta-llama/Llama-3.2-1B.
  If your task is similar to the task the model of the checkpoint was trained on, you can already use LlamaForCausalLM for predictions without further training.
  [INFO|configuration_utils.py:1097] 2025-05-06 22:26:32,860 >> loading configuration file generation_config.json from cache at /home/foremans/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08/generation_config.json
  [INFO|configuration_utils.py:1142] 2025-05-06 22:26:32,860 >> Generate config GenerationConfig {
  "bos_token_id": 128000,
  "do_sample": true,
  "eos_token_id": 128001,
  "temperature": 0.6,
  "top_p": 0.9
  }

  [INFO|trainer.py:698] 2025-05-06 22:26:33,878 >> max_steps is given, it will override any value given in num_train_epochs
  [INFO|trainer.py:748] 2025-05-06 22:26:33,879 >> Using auto half precision backend
  [INFO|trainer.py:2414] 2025-05-06 22:26:52,889 >> ***** Running training *****
  [INFO|trainer.py:2415] 2025-05-06 22:26:52,889 >>   Num examples = 1,920
  [INFO|trainer.py:2416] 2025-05-06 22:26:52,889 >>   Num Epochs = 9,223,372,036,854,775,807
  [INFO|trainer.py:2417] 2025-05-06 22:26:52,889 >>   Instantaneous batch size per device = 8
  [INFO|trainer.py:2420] 2025-05-06 22:26:52,889 >>   Total train batch size (w. parallel, distributed & accumulation) = 192
  [INFO|trainer.py:2421] 2025-05-06 22:26:52,889 >>   Gradient Accumulation steps = 1
  [INFO|trainer.py:2422] 2025-05-06 22:26:52,890 >>   Total optimization steps = 10
  [INFO|trainer.py:2423] 2025-05-06 22:26:52,890 >>   Number of trainable parameters = 1,235,814,400
  [INFO|integration_utils.py:831] 2025-05-06 22:26:52,890 >> Automatic Weights & Biases logging enabled, to disable set os.environ["WANDB_DISABLED"] = "true"
  wandb: WARNING The `run_name` is currently set to the same value as `TrainingArguments.output_dir`. If this was not intended, please specify a different run name by setting the `TrainingArguments.run_name` parameter.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,121 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,121 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [WARNING|_logger.py:68] 2025-05-06 22:26:54,122 >> `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
  [INFO|trainer.py:3984] 2025-05-06 22:27:05,127 >> Saving model checkpoint to trainer_output/checkpoint-10
  [INFO|configuration_utils.py:419] 2025-05-06 22:27:05,143 >> Configuration saved in trainer_output/checkpoint-10/config.json
  [INFO|configuration_utils.py:911] 2025-05-06 22:27:05,150 >> Configuration saved in trainer_output/checkpoint-10/generation_config.json
  [INFO|modeling_utils.py:3572] 2025-05-06 22:27:10,292 >> Model weights saved in trainer_output/checkpoint-10/model.safetensors
  [INFO|tokenization_utils_base.py:2510] 2025-05-06 22:27:10,304 >> tokenizer config file saved in trainer_output/checkpoint-10/tokenizer_config.json
  [INFO|tokenization_utils_base.py:2519] 2025-05-06 22:27:10,312 >> Special tokens file saved in trainer_output/checkpoint-10/special_tokens_map.json
  [INFO|trainer.py:2681] 2025-05-06 22:27:20,107 >>

  Training completed. Do not forget to share your model on huggingface.co/models =)


  [INFO|trainer.py:3984] 2025-05-06 22:27:20,141 >> Saving model checkpoint to trainer_output
  [INFO|configuration_utils.py:419] 2025-05-06 22:27:20,149 >> Configuration saved in trainer_output/config.json
  [INFO|configuration_utils.py:911] 2025-05-06 22:27:20,155 >> Configuration saved in trainer_output/generation_config.json
  [INFO|modeling_utils.py:3572] 2025-05-06 22:27:25,182 >> Model weights saved in trainer_output/model.safetensors
  [INFO|tokenization_utils_base.py:2510] 2025-05-06 22:27:25,191 >> tokenizer config file saved in trainer_output/tokenizer_config.json
  [INFO|tokenization_utils_base.py:2519] 2025-05-06 22:27:25,197 >> Special tokens file saved in trainer_output/special_tokens_map.json
  [INFO|trainer.py:4307] 2025-05-06 22:27:25,394 >>
  ***** Running Evaluation *****
  [INFO|trainer.py:4311] 2025-05-06 22:27:25,395 >>   Num examples: Unknown
  [INFO|trainer.py:4312] 2025-05-06 22:27:25,395 >>   Batch size = 8
  {'loss': 2.847, 'grad_norm': 3.8245272636413574, 'learning_rate': 5e-05, 'epoch': 0.1, 'num_input_tokens_seen': 24576}
  {'loss': 2.9574, 'grad_norm': 7.945530414581299, 'learning_rate': 4.5e-05, 'epoch': 0.2, 'num_input_tokens_seen': 49152}
  {'loss': 3.1086, 'grad_norm': 7.155135631561279, 'learning_rate': 4e-05, 'epoch': 0.3, 'num_input_tokens_seen': 73728}
  {'loss': 2.9751, 'grad_norm': 4.435009956359863, 'learning_rate': 3.5e-05, 'epoch': 0.4, 'num_input_tokens_seen': 98304}
  {'loss': 3.0095, 'grad_norm': 4.177059173583984, 'learning_rate': 3e-05, 'epoch': 0.5, 'num_input_tokens_seen': 122880}
  {'loss': 2.9153, 'grad_norm': 4.262296676635742, 'learning_rate': 2.5e-05, 'epoch': 0.6, 'num_input_tokens_seen': 147456}
  {'loss': 2.8742, 'grad_norm': 6.913131237030029, 'learning_rate': 2e-05, 'epoch': 0.7, 'num_input_tokens_seen': 172032}
  {'loss': 3.2855, 'grad_norm': 5.904435157775879, 'learning_rate': 1.5e-05, 'epoch': 0.8, 'num_input_tokens_seen': 196608}
  {'loss': 2.9934, 'grad_norm': 4.500864028930664, 'learning_rate': 1e-05, 'epoch': 0.9, 'num_input_tokens_seen': 221184}
  {'loss': 2.8064, 'grad_norm': 6.904043197631836, 'learning_rate': 5e-06, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  {'train_runtime': 12.4474, 'train_samples_per_second': 154.249, 'train_steps_per_second': 0.803, 'train_tokens_per_second': 822.661, 'train_loss': 2.977239990234375, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  {'eval_loss': 1.6778849363327026, 'eval_accuracy': 0.6173228346456693, 'eval_runtime': 13.2043, 'eval_samples_per_second': 0.227, 'eval_steps_per_second': 0.076, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  wandb:
  wandb: 🚀 View run cosmic-meadow-38 at: https://wandb.ai/aurora_gpt/ezpz-hf_trainer-meta-llama-Llama-3.2-1B/runs/6yl6uks0
  wandb: Find logs at: ../../../../../../lus/flare/projects/datascience/foremans/projects/saforem2/ezpz/wandb/run-20250506_222620-6yl6uks0/logs
  {'loss': 2.847, 'grad_norm': 3.8245272636413574, 'learning_rate': 5e-05, 'epoch': 0.1, 'num_input_tokens_seen': 24576}
  {'loss': 2.9574, 'grad_norm': 7.945530414581299, 'learning_rate': 4.5e-05, 'epoch': 0.2, 'num_input_tokens_seen': 49152}
  {'loss': 3.1086, 'grad_norm': 7.155135631561279, 'learning_rate': 4e-05, 'epoch': 0.3, 'num_input_tokens_seen': 73728}
  {'loss': 2.9751, 'grad_norm': 4.435009956359863, 'learning_rate': 3.5e-05, 'epoch': 0.4, 'num_input_tokens_seen': 98304}
  {'loss': 3.0095, 'grad_norm': 4.177059173583984, 'learning_rate': 3e-05, 'epoch': 0.5, 'num_input_tokens_seen': 122880}
  {'loss': 2.9153, 'grad_norm': 4.262296676635742, 'learning_rate': 2.5e-05, 'epoch': 0.6, 'num_input_tokens_seen': 147456}
  {'loss': 2.8742, 'grad_norm': 6.913131237030029, 'learning_rate': 2e-05, 'epoch': 0.7, 'num_input_tokens_seen': 172032}
  {'loss': 3.2855, 'grad_norm': 5.904435157775879, 'learning_rate': 1.5e-05, 'epoch': 0.8, 'num_input_tokens_seen': 196608}
  {'loss': 2.9934, 'grad_norm': 4.500864028930664, 'learning_rate': 1e-05, 'epoch': 0.9, 'num_input_tokens_seen': 221184}
  {'loss': 2.8064, 'grad_norm': 6.904043197631836, 'learning_rate': 5e-06, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  {'train_runtime': 27.2171, 'train_samples_per_second': 70.544, 'train_steps_per_second': 0.367, 'train_tokens_per_second': 376.234, 'train_loss': 2.977239990234375, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  ***** train metrics *****
  epoch                    =        1.0
  num_input_tokens_seen    =     245760
  train_loss               =     2.9772
  train_runtime            = 0:00:27.21
  train_samples            =     726000
  train_samples_per_second =     70.544
  train_steps_per_second   =      0.367
  train_tokens_per_second  =    376.234
  {'eval_loss': 1.6778849363327026, 'eval_accuracy': 0.6173228346456693, 'eval_runtime': 7.9617, 'eval_samples_per_second': 0.377, 'eval_steps_per_second': 0.126, 'epoch': 1.0, 'num_input_tokens_seen': 245760}
  ***** eval metrics *****
  epoch                   =        1.0
  eval_accuracy           =     0.6173
  eval_loss               =     1.6779
  eval_runtime            = 0:00:07.96
  eval_samples            =         50
  eval_samples_per_second =      0.377
  eval_steps_per_second   =      0.126
  num_input_tokens_seen   =     245760
  perplexity              =     5.3542
  Application 3917764c resources: utime=2709s stime=1798s maxrss=15499424KB inblock=959040 oublock=38691080 minflt=31555618 majflt=73083 nvcsw=1288306 nivcsw=2040486
  [2025-05-06 22:27:37][I][ezpz/launch:201] Execution finished @ 2025-05-06-222737
  [2025-05-06 22:27:37][I][ezpz/launch:202] Command took 93.85 seconds to run. Exiting.
  took: 0h:01m:45s
  ```

  </details>

## 🏎️ Megatron-DeepSpeed

``` bash
git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
cd Megatron-DeepSpeed
source <(curl -L https://bit.ly/ezpz-utils)
python3 -m pip install -e \
    deepspeed \
    "git+https://github.com/saforem2/ezpz"
bash train_alcf.sh
```

## 🙌 Acknowledgements

> This research used resources of the Argonne Leadership Computing
> Facility, which is a DOE Office of Science User Facility supported
> under Contract DE-AC02-06CH11357.

[^1]: In *general*, you should be wary of running random scripts from
    the internet.

[^2]: <https://bit.ly/ezpz-utils>, since
    <https://raw.githubusercontent.com/saforem2/ezpz/main/bin/utils.sh>
    is a bit of a pain

[^3]: e.g. `${NHOSTS}`, `${NGPU_PER_HOST}`, `${NGPUS}`, …

[^4]: *Should also work with SLURM* (needs further testing)

[^5]: On any of the ALCF systems, including:
    [Aurora](https://alcf.anl.gov/aurora),
    [Polaris](https://alcf.anl.gov/polaris), …, etc.

[^6]: Or, for example, if you would like to exclude a node you suspect
    is having issues

[^7]: You should *always* be working in a virtual environment. See: [🏖️
    Shell Environment](#shell-environment)

[^8]: Will automatically be reported to W&B if a run is detected
