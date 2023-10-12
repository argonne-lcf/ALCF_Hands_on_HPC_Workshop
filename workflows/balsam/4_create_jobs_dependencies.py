from balsam.api import Job

# Create a collection of jobs, each one depending on the job before it
n = 0
job = Job(site_name="polaris_tutorial",
          app_id="Hello",
          workdir=f"demo/hello_deps{n}",
          parameters={"say_hello_to": f"world {n}!"},
          tags={"workflow": "hello_deps"},
          node_packing_count=4,
          gpus_per_rank=1)
job.save()

for n in range(8):
    job = Job(site_name="polaris_tutorial",
              app_id="Hello",
              workdir=f"demo/hello_deps{n}",
              parameters={"say_hello_to": f"world {n}!"},
              tags={"workflow": "hello_deps"},
              node_packing_count=4,
              gpus_per_rank=1,
              parent_ids=[job.id])  # Sets a dependency on the prior job
    job.save()
