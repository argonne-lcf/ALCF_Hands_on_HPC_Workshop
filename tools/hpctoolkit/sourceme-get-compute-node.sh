qsub -I -X -l select=1,walltime=2:00:00,place=scatter -l filesystems=flare -A alcf_training -q alcf_training
