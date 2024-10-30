# Polaris file systems examples

Each of the examples in this directory relies on the 'mpi-io-test' benchmark. To build the benchmark, simply run:

```
cc -o mpi-io-test mpi-io-test.c
```

These examples create user-specific scratch spaces in the Lustre Grand filesytem project directories provided for this workshop.

# Lustre striping example

This example executes the 'mpi-io-test' benchmark using a couple of different Lustre striping configurations:
 - using a stripe count of 1 (default on ALCF Lustre volumes)
 - using a stripe count of 4

The Lustre `lfs setstripe` command is used to change the default stripe settings associated with a file/directory. `lfs getstripe` shows the current striping settings for a file/directory, including the stripe size, count, and the list of Lustre OSTs the file is striped over.

Users should be able to see clear performance discrepancies between the two striping configurations when examining benchmark output.

# SSD stage-out / stage-in example

These examples demonstrate how users can stage data in and out of the node-local SSD storage devices on Polaris.

NOTE: the `ssd-stage-out` example must be ran _before_ the `ssd-stage-in` example, as it writes the data that is used for stage-in.

The `ssd-stage-out` example runs `mpi-io-test` benchmark twice in write-only mode: first against Lustre scratch and next against the local SSD devices. This example then stages the benchmark output to scratch space on Grand so that it can be subsequently used by the `ssd-stage-in` example.

The `ssd-stage-in` example runs `mpi-io-test` benchmark twice in read-only mode: first against Lustre scratch (using the staged out data from above) and next against the local SSD devices (by staging-in the data from above).

Users should be able to notice two key things from the benchmark output:
  1.) Surprisingly, write performance in the `ssd-stage-out` example is comparable for the Lustre and node-local SSD cases.
      - This is likely due to caching effects that allow Lustre writes to complete "locally".
  2.) Read performance in the `ssd-stage-in` example is much higher when using the SSDs (compared to Lustre), as the data must be read from Lustre servers and not a local cache.
