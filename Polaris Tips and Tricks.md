# Polaris Cheat-Sheet

This document is to hold some answers to questions asked on Slack, meant to be easy to find and reuse.

### How do I get an interactive job?

From Colleen on Slack: you use -I (uppercase 'i')  in the qsub command instead of giving it a script. For example, for polaris: 
```bash
qsub -l select=1:system=polaris -l walltime=0:60:00 -l filesystems=home:eagle -q SDL_Workshop -A SDL_Workshop -I  
```
this requests 1 node for 60 min with the home and eagle filesystem

### How do I figure out where my jobs are running? - FROM THE LOGIN NODE

This is, unfortunately, a multistep process.  First, find out what your job ID is, if you don't know it already:

```bash
$ qstat -wa -u $(whoami)

polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov: 
                                                                                                   Req'd  Req'd   Elap
Job ID                         Username        Queue           Jobname         SessID   NDS  TSK   Memory Time  S Time
------------------------------ --------------- --------------- --------------- -------- ---- ----- ------ ----- - -----
337374.polaris-pbs-01.hsn.cm.* cadams          S330213         run_ct_pytorch*    49036  256 16384    --  00:30 R 00:07
```

Your output will be different, this shows I have one job on 256 nodes with job id 337374.  Get the full info of the job like this:

```bash
qstat -f 337374
```

This gives a LONG output if you have 256 nodes like this job.  In shorter jobs, look for the `exec_host` parameter:

```
 exec_host = x3007c0s13b1n0/0*64+x3007c0s19b0n0/0*64+x3007c0s19b1n0/0*64+x30
	07c0s1b0n0/0*64+x3007c0s1b1n0/0*64+x3007c0s25b0n0/0*64+x3007c0s25b1n0/0
	*64+x3007c0s31b0n0/0*64+x3007c0s31b1n0/0*64+x3007c0s37b0n0/0*64+x3007c0
```

This is a list of nodes that your job is using.  You can ssh to any of them directly while your job is running, for example `ssh x3007c0s25b0n0`.

### How do I figure out where my jobs are running? - FROM AN INTERACTIVE NODE DURING A MULTI-NODE JOB

If you are on the interactive node given to you by qsub, you will have an environment variable you can use to dump the list of nodes in your job:

```bash
cat $PBS_NODEFILE
cadams@x3007c0s13b0n0 : ~
$ cat $PBS_NODEFILE
x3007c0s13b0n0.hsn.cm.polaris.alcf.anl.gov
x3111c0s1b0n0.hsn.cm.polaris.alcf.anl.gov
```

You can `ssh` to these nodes from the login nodes or from other compute nodes.  You can use the full hostname, or if you're already on polaris you can use just the first part of the name like so: `ssh x3111c0s1b0n0`.
