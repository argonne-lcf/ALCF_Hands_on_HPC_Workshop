from balsam.api import Job,Site,BatchJob

jobs = [
    Job( site_name="polaris-testing", 
         app_id="Hello", 
         workdir=f"demo/hello_multi{n}", 
         parameters={"say_hello_to": f"world {n}!"},
         tags={"workflow":"hello_multi"}, 
         node_packing_count=4,  # run 4 jobs per node
         gpus_per_rank=1)       # one GPU tile per rank    
    for n in range(8)
]

# Create all n jobs in one call; the list of created jobs is returned
jobs = Job.objects.bulk_create(jobs)

# Get the id for your site
site = Site.objects.get("polaris-testing")

# Create a BatchJob to run jobs with the workflow=hello_multi tag
BatchJob.objects.create(
    num_nodes=1,
    wall_time_min=10,
    queue="debug",
    project="datascience",
    site_id=site.id,
    filter_tags={"workflow":"hello_multi"},
    job_mode="mpi"
)
