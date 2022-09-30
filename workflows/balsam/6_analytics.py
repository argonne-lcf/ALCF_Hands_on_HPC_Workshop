#!/usr/bin/env python
from balsam.api import models
from balsam.api import EventLog,Job
from balsam.analytics import throughput_report
from balsam.analytics import utilization_report
from matplotlib import pyplot as plt

# Fetch jobs and events for the Hello app
app = models.App.objects.get(site_name="thetagpu_tutorial",name="Hello")
jl = Job.objects.filter(app_id=app.id)
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
