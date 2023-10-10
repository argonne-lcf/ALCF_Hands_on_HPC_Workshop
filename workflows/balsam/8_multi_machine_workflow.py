from balsam.api import Job,BatchJob,Site
import time

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
    print("Waiting for Polaris to finish")
    time.sleep(5)
    if Job.objects.get(id=polaris_job.id).state == "JOB_FINISHED":
        print("Polaris has finished! Starting ThetaGPU job")
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
