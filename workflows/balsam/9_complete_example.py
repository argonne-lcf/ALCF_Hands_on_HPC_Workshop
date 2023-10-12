from balsam.api import ApplicationDefinition, Site, Job, BatchJob

site_name = "polaris_tutorial"
site = Site.objects.get(site_name)


# Define an App that wraps around the compilied executable hello_affinity
# It also loads a module before the app execution and adds the executable path to PATH
# In this case where the application is using all the GPUs of a node,
# we will wrap the executable with a gpu affinity script that places ranks on gpus
class HelloAffinity(ApplicationDefinition):
    site = "polaris_tutorial"

    def shell_preamble(self):
        return '''
            module load PrgEnv-nvhpc
            export PATH=$PATH:/eagle/fallwkshp23/workflows/affinity_gpu/
        '''

    command_template = "set_affinity_gpu_polaris.sh hello_affinity"


HelloAffinity.sync()

# Create 8 jobs running the HelloAffinity app,
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
