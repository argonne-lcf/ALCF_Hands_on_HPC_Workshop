#!/bin/bash   

# only output messages from rank 0
if [[ $PMIX_RANK -eq 0 ]]; then
  $*
else
  $* >& /dev/null
fi
