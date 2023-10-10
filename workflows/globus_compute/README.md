Globus Compute: Remote execution of applications with Globus
===============================================

This tutorial demonstrates how to run applications on Polaris using Globus Compute.  Globus Compute (formerly called FuncX) uses the Globus service to communicate with a user process called an endpoint running on a remote machine.  The Globus Compute endpoint uses Parsl locally to execute work through the machine's scheduler.

Globus Compute can be used to execute functions remotely as a service and can be integrated with Globus Flows to create workflows that automate the inegration of data transfers and function execution.

There are x requirements to deploying an application through Globus Compute on Polaris.
1. Install required globus packages on in environments on Polaris and the remote machine deploying the work.
2. An active Globus Compute endpoint on a Polaris login node.
3. An http connection on the remote machine deploying work.

# Setup

## Installing Globus modules

## Creating and Starting an Endpoint

Login to Polaris and clone this repo.  Activate your environment.
```
source activate /eagle/fallwkshp23/workflows/env/bin/activate
```

Use the sample config polaris_config.yaml provided to configure and start your endpoint.
```
globus-compute-endpoint configure --endpoint-config ./polaris_config.yaml workshop-endpoint
globus-compute-endpoint start workshop-endpoint
```

Verify that your endpoint is active.
```
globus-compute-endpoint list
```

Your endpoint will have and id, copy this unique id, you will need it on your local machine.

You can also verify that your endpoint is communicating with the Globus Service by looking at https://app.globus.org/compute.


