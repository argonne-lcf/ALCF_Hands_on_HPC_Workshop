# ALCF Hands-on HPC Workshop 2023 Darshan hands-on I/O exercises and reference material

## Initial setup

* Log on to ALCF Polaris
* Download the hands-on materials to your home directory.
  * `git clone https://github.com/argonne-lcf/ALCF_Hands_on_HPC_Workshop.git`
  * `cd darshan-hands-on`
* Set up your environment to have access to the utilities needed for the hands-on exercises
  * `source ./polaris-setup-env.sh`

## Running hands-on example programs

* Compile example programs and submit into the job queue (see below for
details on specific example programs)
  * `cc <exampleprogram>.c -o <exampleprogram>`
  * `qsub ./<exampleprogram>.qsub`
* Check the queue to see when your jobs complete
  * `qstat |grep <username>`
* Look for log files in `/lus/grand/logs/darshan/polaris/2023/10/12/<username>*` (or whatever the current day is in GMT)
  * Copy log files to your home directory
* Use the PyDarshan job summary tool or `darshan-parser` to investigate Darshan characterization data
  * `python -m darshan summary <log_path>` command will produce \*.html files with an analysis summary
  * You can use scp to copy these to your laptop to view them in a browser

## Hands-on exercise: helloworld

The hands-on material includes an example application called `helloworld`.
Compile it, run it, and generate the Darshan job summary following the
instructions above.  How many files did the application open?  How much data
did it read, and how much data did it write?  What approximate I/O
performance did it achieve?

## Hands-on exercise: warpdrive

_NOTE: this exercise likely requires some understanding of MPI-IO to complete._

The hands-on material includes an example application called `warpdrive`.
There are two versions of this application: warpdriveA and warpdriveB.  Both
of them do the same amount of I/O from each process, but one of them performs
better than the other.  Which one has the fastest I/O?  Why?

## Hands-on exercise: fidgetspinner

_NOTE: this exercise likely requires some understanding of MPI-IO to complete._

The hands-on material includes an example application called
`fidgetspinner`.  There are two versions of this application:
fidgetspinnerA and fidgetspinnerB.  Both of them do the same amount of
I/O from each process, but one of them performs better than the other.
Which one has the fastest I/O?  Why?

