Balsam Workflow Tool for HPC
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
## Getting started with Balsam

To work through this tutorial (or any Balsam functionality), you first need to get a login to the Balsam server.  To get a login, email the ALCF Help Desk.

## Getting started on Polaris

```bash
# Log in to Polaris
ssh polaris.alcf.anl.gov

# clone this repo
git clone git@github.com:argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
cd ALCF_Hands_on_HPC_Workshop/workflows/balsam
```

## Set up environment

### Workshop setup
During this workshop, you can use a python virtual environment that has all the necessary modules installed.
```shell
source /eagle/fallwkshp23/workflows/env/bin/activate
```

### Creating your own setup
```shell
# Create a virtual environment
module load conda
conda activate base
python -m venv env
source env/bin/activate
pip install --upgrade pip
# We need matplotlib for one of the examples
pip install matplotlib

# Install Balsam
pip install --pre balsam
```

## Set up Balsam

Login to the Balsam server. This will prompt you to visit an ALCF login page; this command will block until you've logged in on the webpage.
```shell
balsam login
```

Create a Balsam site
Note: The "-n" option specifies the site name; the last argument specifies the directory name    

```shell       
balsam site init -n polaris_tutorial polaris_tutorial
cd polaris_tutorial
balsam site start
cd ..
```

## Create a Hello application in Balsam (hello.py)
We define an application by wrapping the command line in a small amount of Python code. Note that the command line is directly represented, and the say_hello_to parameter will be supplied when a job uses this application. Note that we print the available GPUs; this is just to highlight GPU scheduling.

```python
from balsam.api import ApplicationDefinition

class Hello(ApplicationDefinition):
    site = "polaris_tutorial"
    command_template = "echo Hello, {{ say_hello_to }}! CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
Hello.sync()
```

To add the Hello application to the Balsam site, we run the hello.py file
```bash
python hello.py
```


## Create a Hello job using the Balsam CLI interface (1_create_job.sh)
We create a job using the command line interface, providing the say_hello_to parameter, and then list Balsam jobs with the related tag.
```bash
#!/bin/bash -x 

# List apps
balsam app ls --site polaris_tutorial

# Create a Hello job
# Note: tag it with the key-value pair workflow=hello for easy querying later
balsam job create --site polaris_tutorial --app Hello --workdir=demo/hello --param say_hello_to=world --tag workflow=hello --yes

# The job resides in the Balsam server now; list the job
balsam job ls --tag workflow=hello
```

## Submit a BatchJob to run the Hello job (2_submit_batchjob.sh)
We submit a batch job via the command line, and then list BatchJobs and the status of the job. Following a short delay, Balsam will submit the job to PBS, and the job will appear in the usual `qstat` output. The job status command can be run periodically to monitor the job.
```bash
#!/bin/bash

# Submit a batch job to run this job on Polaris
# Note: the command-line parameters are similar to scheduler command lines
# Note: this job will run only jobs with a matching tag
balsam queue submit \
    -n 1 -t 10 -q fallws23single -A fallwkshp23 \
    --site polaris_tutorial \
    --tag workflow=hello \
    --job-mode mpi
  
# List the Balsam BatchJob
# Note: Balsam will submit this job to Cobalt, so it will appear in qstat output after a short delay
balsam queue ls 

# List status of the Hello job
balsam job ls --tag workflow=hello
```

## Create a collection of Hello jobs using the Balsam Python API (3_create_multiple_jobs.py)
We create a collection of jobs, one per GPU on a Polaris node. Though this code doesn't use the GPU, your code will; in the face of a larger collection of Balsam jobs and imbalance in runtimes, Balsam will target GPUs as they become available by setting CUDA_VISIBLE_DEVICES appropriately; the Hello app echoes this variable to illustrate GPU assignment.

```python
from balsam.api import Job

jobs = [
    Job( site_name="polaris_tutorial", 
         app_id="Hello", 
         workdir=f"demo/hello_multi{n}", 
         parameters={"say_hello_to": f"world {n}!"},
         tags={"workflow":"hello_multi"}, 
         node_packing_count=4,  # run 4 jobs per node
         gpus_per_rank=1)       # one GPU per rank    
    for n in range(4)
]

# Create all n jobs in one call; the list of created jobs is returned
jobs = Job.objects.bulk_create(jobs)
```

## Create a collection of Hello jobs with dependencies (4_create_job_dependencies.py)
Similar to the example above, we create a collection of jobs but this time with inter-job dependencies: a simple linear workflow. The jobs will run in order, honoring the job dependency setting (even though the jobs could all run simultaneously). In this case, we use the Python API to define jobs. 

```python
from balsam.api import Job

# Create a collection of jobs, each one depending on the job before it
n=0
job = Job( site_name="polaris_tutorial",
           app_id="Hello", 
           workdir=f"demo/hello_deps{n}", 
           parameters={"say_hello_to": f"world {n}!"},
           tags={"workflow":"hello_deps"}, 
           node_packing_count=4, 
           gpus_per_rank=1)
job.save()
for n in range(1,8):
    job = Job( site_name="polaris_tutorial", 
               app_id="Hello", 
               workdir=f"demo/hello_deps{n}", 
               parameters={"say_hello_to": f"world {n}!"},
               tags={"workflow":"hello_deps"}, 
               node_packing_count=4, 
               gpus_per_rank=1, 
               parent_ids=[job.id])  # Sets a dependency on the prior job
    job.save()
```

## Create a Batch Job with the python API (5_submit_batchjob.py)
This script uses the python API to create a Batch Job on the Polaris scheduler.  This Batch Job will only run jobs with the tag `workflow=hello_deps`.


```python
from balsam.api import BatchJob,Site

# Get the id for your site
site = Site.objects.get("polaris_tutorial")

# Create a BatchJob to run jobs with the workflow=hello_deps tag
BatchJob.objects.create(
    num_nodes=1,
    wall_time_min=10,
    queue="fallws23single",
    project="fallwkshp23",
    site_id=site.id,
    filter_tags={"workflow":"hello_deps"},
    job_mode="mpi"
)
```

## Use the Python API to examine job progress (6_examine_timestamps.py)
As Balsam runs jobs, they advance through states that include transferring input files, job submission, execution, and transferring output files. Users can monitor these events programmatically, to build dynamic workflows or to monitor throughput of their jobs.

```python
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

## Making plots of utilization and throughput (7_analytics_example.py)
This example will query the `Hello` jobs run with the tag `workflow=hello_deps`, to produce plots of utilization and throughput.

```python
from balsam.api import models
from balsam.api import EventLog,Job
from balsam.analytics import throughput_report
from balsam.analytics import utilization_report
from matplotlib import pyplot as plt

# Fetch jobs and events for the Hello app
app = models.App.objects.get(site_name="polaris_tutorial",name="Hello")
jl = Job.objects.filter(app_id=app.id,tags={"workflow":"hello_deps"})

events = EventLog.objects.filter(job_id=[job.id for job in jl])

# Generate a throughput report
times, done_counts = throughput_report(events, to_state="JOB_FINISHED")

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
fig = plt.Figure()
plt.step(elapsed_minutes, done_counts, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Jobs completed")
plt.savefig('throughput.png')

# Generate a utilization report
plt.figure()
times, util = utilization_report(events, node_weighting=True)

t0 = min(times)
elapsed_minutes = [(t - t0).total_seconds() / 60 for t in times]
plt.step(elapsed_minutes, util, where="post")
plt.xlabel("Elapsed time (minutes)")
plt.ylabel("Utilization")
plt.savefig("utilization.png")
```

## Supplemental: Create jobs at multiple sites (8_multi_machine_workflow.sh)
With Sites established on multiple machines, we can submit jobs to multiple sites from wherever Balsam is installed: here on Polaris, or on your laptop. This example creates jobs at ThetaGPU and Polaris, and submits batch jobs to run them. A job dependency is created between the Polaris job and the ThetaGPU job; the ThetaGPU job will only complete after the Polaris job completes. 

```python
#!/usr/bin/env python
from balsam.api import Job,BatchJob,Site

# Create a collection of jobs, each one depending on the job before it

polaris_job = Job( site_name="polaris_tutorial",
           app_id="Hello", 
           workdir=f"demo/multi_machine", 
           parameters={"say_hello_to": f"Polaris"},
           tags={"workflow":"multi_machine"}, 
           node_packing_count=4, 
           gpus_per_rank=1)
polaris_job.save()

thetagpu_job = Job( site_name="thetagpu_tutorial",
                    app_id="Hello", 
                    workdir=f"demo/multi_machine", 
                    parameters={"say_hello_to": f"ThetaGPU"},
                    tags={"workflow":"multi_machine"}, 
                    node_packing_count=4, 
                    gpus_per_rank=1,
                    parent_ids=[polaris_job.id])
thetagpu_job.save()

# Get the ids for your sites
polaris_site = Site.objects.get("polaris_tutorial")
thetagpu_site = Site.objects.get("thetagpu_tutorial")

# Create a BatchJob to run jobs on Polaris
BatchJob.objects.create(
    num_nodes=1,
    wall_time_min=5,
    queue="fallws23single",
    project="fallwkshp23",
    site_id=polaris_site.id,
    filter_tags={"workflow":"multi_machine"},
    job_mode="mpi"
)

# Create a BatchJob to run jobs on ThetaGPU once the polaris job finishes
polaris_job_unfinished = True
while polaris_job_unfinished:
    if Job.objects.get(id=polaris_job.id).state == "JOB_FINISHED":
        polaris_job_unfinished = False
        BatchJob.objects.create(
            num_nodes=1,
            wall_time_min=10,
            queue="single-gpu",
            project="datascience",
            site_id=thetagpu_site.id,
            filter_tags={"workflow":"multi_machine"},
            job_mode="mpi"
        )
```
## A Complete Example for Running Multi-node MPI Jobs on Polaris (9_complete_example.py)

This example runs instances of [hello_affinity](https://github.com/argonne-lcf/GettingStarted/tree/master/Examples/Polaris/affinity_gpu) on 2 Polaris nodes with 4 ranks per node.

```python
from balsam.api import ApplicationDefinition, Site, Job, BatchJob

site_name = "polaris_tutorial"
site = Site.objects.get(site_name)


# Define the App that wraps around a compiled executable
# It also loads a module before the app execution
# In this case where the application is using all the GPUs of a node,
# we will wrap the executable with a gpu affinity script
class HelloAffinity(ApplicationDefinition):
    site = "polaris_tutorial"

    def shell_preamble(self):
        return '''
            module load PrgEnv-nvhpc
            export PATH=$PATH:/home/csimpson/polaris/GettingStarted/Examples/Polaris/affinity_gpu/
        '''

    command_template = "set_affinity_gpu_polaris.sh hello_affinity"


HelloAffinity.sync()

# Create 8 jobs running that app,
# 2 nodes per job, 4 ranks per node, 1 gpu per rank
for i in range(8):
    Job.objects.create( app_id="HelloAffinity",
                        site_name="polaris_tutorial",
                        workdir=f"affinity/{i}",
                        gpus_per_rank=1,
                        threads_per_rank=16,
                        threads_per_core=2,
                        ranks_per_node=4,
                        node_packing_count=1,
                        num_nodes=2,
                        launch_params={"cpu_bind": "depth"},
                        tags={"test": "affinity"},)

# Run the jobs by creating a batch job
BatchJob.objects.create(
    site_id=site.id,
    num_nodes=4,
    wall_time_min=10,
    filter_tags={"test": "affinity"},
    job_mode="mpi",
    queue="fallws23scaling",
    project="fallwkshp23",)
```

## Next Steps 
The examples above demonstrate the basic functionality of Balsam: defining applications, describing jobs, submitting batch jobs to run those jobs, and monitoring throughput, on one site or across multiple sites. With these capabilities, we can define more complex workflows that dynamically adapt as jobs run. For a more detailed application, see the [hyperparameter optimization example](https://github.com/argonne-lcf/balsam_tutorial/tree/main/hyperopt) in the ALCF github repository or the [LAMMPS example](https://github.com/CrossFacilityWorkflows/DOE-HPC-workflow-training/tree/main/Balsam/ALCF) from the recent DOE Joint-lab workflows workshop.

