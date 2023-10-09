#!/bin/bash

# Create jobs at three sites

echo ThetaGPU
balsam job create --site thetagpu_tutorial --app VecNorm --workdir multisite/thetagpu --param vec=[3,4] --tag workflow=vec_multisite --yes

echo Polaris
balsam job create --site polaris-testing --app VecNorm --workdir multisite/sunspot --param vec=[3,4] --tag workflow=vec_multisite --yes

# List the jobs
balsam job ls --tag workflow=vec_multisite
