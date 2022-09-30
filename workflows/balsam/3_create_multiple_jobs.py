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
