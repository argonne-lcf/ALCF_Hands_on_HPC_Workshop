#!/bin/bash

# Create jobs at four sites
# echo ThetaKNL
# balsam job create --site thetaknl_tutorial --app Hello --workdir multisite/thetaknl --param say_hello_to=thetaknl --tag workflow=hello_multisite --yes

# echo Cooley
# balsam job create --site cooley_tutorial --app Hello --workdir multisite/cooleylogin2 --param say_hello_to=cooleylogin2 --tag workflow=hello_multisite --yes

# echo ThetaGPU
balsam job create --site thetagpu_tutorial --app Hello --workdir multisite/thetagpu --param say_hello_to=thetagpu --tag workflow=hello_multisite --yes

echo Laptop
balsam job create --site /Users/csimpson/my-site --app Hello --workdir multisite/christine_laptop --param say_hello_to=christine_laptop --tag workflow=hello_multisite --yes

echo Polaris
balsam job create --site polaris_tutorial --app Hello --workdir multisite/polaris --param say_hello_to=polaris --tag workflow=hello_multisite --yes

# List the jobs
balsam job ls --tag workflow=hello_multisite
