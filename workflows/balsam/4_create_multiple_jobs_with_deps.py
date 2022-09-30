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
    job_mode="mpi"
)
