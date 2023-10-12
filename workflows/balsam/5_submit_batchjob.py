from balsam.api import BatchJob, Site

# Get the id for your site
site = Site.objects.get("polaris_tutorial")

# Create a BatchJob to run jobs with the workflow=hello_deps tag
BatchJob.objects.create(
    num_nodes=1,
    wall_time_min=5,
    queue="fallws23single",
    project="fallwkshp23",
    site_id=site.id,
    filter_tags={"workflow": "hello_deps"},
    job_mode="mpi"
)
