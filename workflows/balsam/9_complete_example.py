from balsam.api import ApplicationDefinition,Site,Job,BatchJob

site_name = "polaris-testing"
site = Site.objects.get(site_name)

# Define the App
class Hello(ApplicationDefinition):
    site = site_name
   
   # The shell preamble allows you to execute shell commands prior to the job running
    def shell_preamble(self):
        return'''
            source /home/csimpson/polaris/setup.sh
        '''

    # The command template is your executable
    command_template = 'echo "Hello Polaris CUDA device "$CUDA_VISIBLE_DEVICES'

Hello.sync() 

# Create 8 jobs running that app, 1 gpu per rank, 4 ranks per node
for i in range(8):
  Job.objects.create(app_id="Hello",
                    site_name=site_name,
                    workdir=f"test/{i}",
                    gpus_per_rank=1,
                    ranks_per_node=4,
                    node_packing_count=1,
                    num_nodes=1,
                    tags={"test":"hello"},
                    )

# Run the jobs by creating a batch job
BatchJob.objects.create(
    site_id=site.id,
    num_nodes=1,
    wall_time_min=10,
    filter_tags={"test":"hello"},
    job_mode="mpi",
    queue="debug",  
    project="datascience",  
)
  
