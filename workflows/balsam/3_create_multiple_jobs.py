from balsam.api import Job

# Define 8 Balsam jobs calling app Hello, run each Hello instance on 1 GPU
jobs = [
    Job(site_name="polaris_tutorial",
        app_id="Hello",
        workdir=f"demo/hello_multi{n}",
        parameters={"say_hello_to": f"world {n}!"},
        tags={"workflow": "hello_multi"},
        node_packing_count=4,  # run 4 jobs per node
        gpus_per_rank=1,)       # one GPU per rank/job
    for n in range(8)
]

# Create all n jobs in one call; the list of created jobs is returned
jobs = Job.objects.bulk_create(jobs)
