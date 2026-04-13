qsub -I -X -l select=1,walltime=1:00:00,place=scatter -l filesystems=flare -A gpu_hack -q debug
