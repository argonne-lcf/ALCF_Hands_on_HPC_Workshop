#!/usr/bin/env python
from datetime import datetime,timedelta
from balsam.api import EventLog

for evt in EventLog.objects.filter(tags={"workflow": "hello_multi"}):
    print("Job:",evt.job_id)  # Job ID
    print(evt.timestamp)      # Time of state change (UTC)
    print(evt.from_state)     # From which state the job transitioned
    print(evt.to_state)       # To which state
    print(evt.data)           # optional payload
