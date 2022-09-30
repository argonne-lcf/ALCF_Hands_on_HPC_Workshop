Balsam: ALCF Computational Performance Workshop
===============================================

[Balsam](https://github.com/argonne-lcf/balsam) is a toolkit for describing and managing large-scale computational campaigns on supercomputers. The command line interface and Python API make it easy for users to adopt: after wrapping the command line in a few lines of Python code, users describe jobs with accompanying command-line options, which are stored persistently in the Balsam database. A Balsam site runs on a login node or service node of a cluster. When the user submits a batch job through Balsam, the Site pulls jobs from the database and executes them within the batch job, achieving high throughput while incurring only a single wait-time in the queue.

The full [Balsam documentation](https://balsam.readthedocs.io/en/latest) covers all functionality for users, including additional examples, and describes the Balsam architecture for potential developers; contributions welcome! 

Let's put some useful Balsam command lines here at the top for reference. Note that some of these commands have additional options; you can always see these by appending `--help`.
```
# List your sites
balsam site ls

# List your apps
balsam app ls

# List your jobs (specify tag for expedience)
balsam job ls --tag workflow=hello

# List your batch jobs
balsam queue ls
```


## Getting started on ThetaGPU

```bash
# Log in to Theta
ssh theta.alcf.anl.gov

# clone this repo
git clone https://github.com/argonne-lcf/CompPerfWorkshop.git
cd CompPerfWorkshop/00_workflows/balsam_demo
```

## Set up Balsam and supporting codes
```
# Create a virtual environment
/lus/theta-fs0/software/datascience/conda/2021-09-22/mconda3/bin/python -m venv env
source env/bin/activate
pip install --upgrade pip
# We need matplotlib for one of the examples
pip install matplotlib

# Install Balsam
pip install balsam

# Login to the Balsam server. This will prompt you to visit an ALCF login page; this command will
# block until you've logged in on the webpage.
balsam login

# Load the cobalt-gpu module, so Balsam submits to the ThetaGPU
module load cobalt/cobalt-gpu

# Create a Balsam site
# Note: The "-n" option specifies the site name; the last argument specifies the directory name                              
balsam site init -n thetagpu_tutorial thetagpu_tutorial
cd thetagpu_tutorial
balsam site start
cd ..
```

## Create a Hello application in Balsam (hello.py)
We define an application by wrapping the command line in a small amount of Python code. Note that the command line is directly represented, and the say_hello_to parameter will be supplied when a job uses this application. Note that we print the available GPUs; this is just to highlight GPU scheduling.

```python
from balsam.api import ApplicationDefinition

class Hello(ApplicationDefinition):
    site = "thetagpu_tutorial"
    command_template = "echo Hello, {{ say_hello_to }}! CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
Hello.sync()
```

To add the Hello application to the Balsam site, we run the hello.py file
```bash
python hello.py
```


## Create a Hello job using the Balsam CLI interface (1_create_job.sh)
We create a job using the command line interface, providing the say_hello_to parameter, and then list Balsam jobs with the related tag.
```python
#!/bin/bash -x 

# Create a Hello job
# Note: tag it with the key-value pair workflow=hello for easy querying later
balsam job create --site thetagpu_tutorial --app Hello --workdir=demo/hello --param say_hello_to=world --tag workflow=hello --yes

# The job resides in the Balsam server now; list the job
balsam job ls --tag workflow=hello
```

## Submit a BatchJob to run the Hello job (2_submit_batchjob.sh)
We submit a batch job via the command line, and then list BatchJobs and the status of the job. Following a short delay, Balsam will submit the job to Cobalt, and the job will appear in the usual `qstat` output. The job status command can be run periodically to monitor the job.
```bash
#!/bin/bash

# Submit a batch job to run this job on ThetaGPU
# Note: the command-line parameters are similar to scheduler command lines
# Note: this job will run only jobs with a matching tag
balsam queue submit \
    -n 1 -t 10 -q training-gpu -A training-gpu \
    --site thetagpu_tutorial \
    --tag workflow=hello \
    --job-mode mpi
  
# List the Balsam BatchJob
# Note: Balsam will submit this job to Cobalt, so it will appear in qstat output after a short delay
balsam queue ls 

# List status of the Hello job
balsam job ls --tag workflow=hello
```

## Create a collection of Hello jobs using the Balsam Python API (3_create_multiple_jobs.py)
We create a collection of jobs, one per GPU on a ThetaGPU node. Though this code doesn't use the GPU, your code will; in the face of a larger collection of Balsam jobs and imbalance in runtimes, Balsam will target GPUs as they become available by setting CUDA_VISIBLE_DEVICES appropriately; the Hello app echoes this variable to illustrate GPU assignment.

```python
#!/usr/bin/env python
from balsam.api import Job

jobs = [
    Job( site_name="thetagpu_tutorial", 
         app_id="Hello", 
         workdir=f"demo/hello_multi{n}", 
         parameters={"say_hello_to": f"world {n}!"},
         tags={"workflow":"hello_multi"}, 
         node_packing_count=8,  # run 8 jobs per node
         gpus_per_rank=1)       # one GPU per rank    
    for n in range(8)
]

# Create all n jobs in one call; the list of created jobs is returned
jobs = Job.objects.bulk_create(jobs)

```

## Create a collection of Hello jobs with dependencies (4_create_multiple_jobs_with_deps.py)
Similar to the example above, we create a collection of jobs but this time with inter-job dependencies: a simple linear workflow. The jobs will run in order, honoring the job dependency setting (even though the jobs could all run simultaneously). In this case, we use the Python API to define jobs and batch jobs. 

```python
#!/usr/bin/env python
from balsam.api import Job,BatchJob,Site

# Create a collection of jobs, each one depending on the job before it
n=0
job = Job( site_name="thetagpu_tutorial",
           app_id="Hello", 
           workdir=f"demo/hello_deps{n}", 
           parameters={"say_hello_to": f"world {n}!"},
           tags={"workflow":"hello_deps"}, 
           node_packing_count=8, 
           gpus_per_rank=1)
job.save()
for n in range(1,8):
    job = Job( site_name="thetagpu_tutorial", 
               app_id="Hello", 
               workdir=f"demo/hello_deps{n}", 
               parameters={"say_hello_to": f"world {n}!"},
               tags={"workflow":"hello_deps"}, 
               node_packing_count=8, 
               gpus_per_rank=1, 
               parent_ids=[job.id])  # Sets a dependency on the prior job
    job.save()
    
# Get the id for your site
site = Site.objects.get("thetagpu_tutorial")

# Create a BatchJob to run jobs with the workflow=hello_deps tag
BatchJob.objects.create(
    num_nodes=1,
    wall_time_min=10,
    queue="training-gpu",
    project="Comp_Perf_Workshop",
    site_id=site.id,
    filter_tags={"workflow":"hello_deps"},
    job_mode="mpi",
)
```

## Use the Python API to monitor jobs (5_monitor_jobs.py)
As Balsam runs jobs, they advance through states that include transferring input files, job submission, execution, and transferring output files. Users can monitor these events programmatically, to build dynamic workflows or to monitor throughput of their jobs.

```python
#!/usr/bin/env python
from datetime import datetime,timedelta
from balsam.api import EventLog

# Fetch events that occurred in the past 24 hours
yesterday = datetime.utcnow() - timedelta(days=1)
for evt in EventLog.objects.filter(timestamp_after=yesterday):
    print("Job:",evt.job_id)  # Job ID
    print(evt.timestamp)      # Time of state change (UTC)
    print(evt.from_state)     # From which state the job transitioned
    print(evt.to_state)       # To which state
    print(evt.data)           # optional payload
```


## The Python API includes analytics support for utilization and throughput (6_analytics.py)
This example will query the Hello jobs run to this point, to produce plots of utilization and throughput.

```python
#!/usr/bin/env python
from balsam.api import models
from balsam.api import EventLog,Job
from balsam.analytics import throughput_report
from balsam.analytics import utilization_report
from matplotlib import pyplot as plt

# Fetch jobs and events for the Hello app
app = models.App.objects.get(site_name="thetagpu_tutorial",name="Hello")
jl = Job.objects.filter(app_id=app.id)
events = EventLog.objects.filter(job_id=[job.id for job in jl])

# Generate a throughput report
times, done_counts = throughput_report(events, to_state="JOB_FINISHED")

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
fig = plt.Figure()
plt.step(elapsed_minutes, done_counts, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Jobs completed")
plt.savefig("throughput.png")

# Generate a utilization report
times, util = utilization_report(events, node_weighting=True)

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
plt.step(elapsed_minutes, util, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Utilization")
plt.savefig("utilization.png")
```

## Supplemental: Create a collection of jobs at multiple sites (7_create_jobs_at_multiple_sites.sh)
With Sites established on multiple machines, we can submit jobs to multiple sites from wherever Balsam is installed: here on Theta, or on your laptop. This example creates jobs at four different sites (including my laptop), and submits batch jobs to run them. As these jobs run, they can be monitored using `balsam job ls --tag workflow=hello_multisite --site all`. 

> **⚠ WARNING: Extra Setup Required **  
> This example will only work for you if you replicate this site/app setup
```python
#!/bin/bash

# Create jobs at four sites
echo ThetaKNL
balsam job create --site thetaknl_tutorial --app Hello --workdir multisite/thetaknl --param say_hello_to=thetaknl --tag workflow=hello_multisite --yes

echo Cooley
balsam job create --site cooley_tutorial --app Hello --workdir multisite/cooleylogin2 --param say_hello_to=cooleylogin2 --tag workflow=hello_multisite --yes

echo ThetaGPU
balsam job create --site thetagpu_tutorial --app Hello --workdir multisite/thetagpu --param say_hello_to=thetagpu --tag workflow=hello_multisite --yes

echo Laptop
balsam job create --site tom_laptop --app Hello --workdir multisite/tom_laptop --param say_hello_to=tom_laptop --tag workflow=hello_multisite --yes

# List the jobs
balsam job ls --tag workflow=hello_multisite
```


Create a collection of jobs across Sites...
```bash
#!/bin/bash

# create jobs at four sites
balsam job create --site thetagpu_tutorial --app Hello --workdir multisite/thetaknl --param say_hello_to=thetaknl --tag workflow=hello_multisite --yes
balsam job create --site thetaknl_tutorial --app Hello --workdir multisite/thetagpu --param say_hello_to=thetagpu --tag workflow=hello_multisite --yes
balsam job create --site cooley_tutorial --app Hello --workdir multisite/cooleylogin2 --param say_hello_to=cooleylogin2 --tag workflow=hello_multisite --yes
balsam job create --site tom_laptop --app Hello --workdir multisite/tom_laptop --param say_hello_to=tom_laptop --tag workflow=hello_multisite --yes

# list the jobs
balsam job ls --tag workflow=hello_multisite
```
...and submit BatchJobs to run them
```bash
#!/bin/bash

# Submit BatchJobs at multiple sites
# theta gpu
balsam queue submit \
  -n 1 -t 10 -q single-gpu -A training-gpu \
  --site thetagpu_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# theta knl
balsam queue submit \
  -n 1 -t 10 -q debug-flat-quad -A training-gpu \
  --site thetaknl_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# cooley
balsam queue submit \
  -n 1 -t 10 -q debug -A training-gpu \
  --site cooley_tutorial \
  --tag workflow=hello_multi \
  --job-mode mpi

# tom laptop
balsam queue submit \
  -n 1 -t 10 -q local -A local \
  --site tom_laptop \
  --tag workflow=hello_multi \
  --job-mode mpi \

# List queues
balsam queue ls
```

## Next Steps 
The examples above demonstrate the basic functionality of Balsam: defining applications, describing jobs, submitting batch jobs to run those jobs, and monitoring throughput, on one site or across multiple sites. With these capabilities, we can define more complex workflows that dynamically adapt as jobs run. For a more detailed application, see the [hyperparameter optimization example](https://github.com/argonne-lcf/balsam_tutorial/tree/main/hyperopt) in the ALCF github repository.

