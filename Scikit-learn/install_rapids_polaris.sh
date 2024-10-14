#!/bin/bash

# install_rapids_polaris.sh

# Install RAPIDS on Polaris
# [check here for the latest version](https://rapids.ai/start.html)

SYSTEM="polaris"
RAPIDS_VERSION=24.08
CUDATOOLKIT_VERSION=12.0
PYTHON_VERSION=3.10
ENV_PATH="/path/to/conda/dir"
BASE_CONDA=2024-04-29
RAPIDS_WORKDIR='/lus/grand/projects/datascience/rapids/polaris/rapids-24.08_polaris && $@'

module load conda/${BASE_CONDA} && \
conda create -y -p ${ENV_PATH}/rapids-${RAPIDS_VERSION}_${SYSTEM} \
-c rapidsai -c nvidia -c conda-forge rapids=${RAPIDS_VERSION} \
python=${PYTHON_VERSION} cudatoolkit=${CUDATOOLKIT_VERSION} \
ipykernel jupyterlab-nvdashboard dask-labextension && \
conda activate ${ENV_PATH}/rapids-${RAPIDS_VERSION}_${SYSTEM} && \
jupyter serverextension enable --py --sys-prefix dask_labextension && \
env=$(basename `echo $CONDA_PREFIX`) && \
python -m ipykernel install --user --name "$env" --display-name "Python [conda env:"$env"]"

cat > activate_rapids_env_polaris.sh << EOF
#!/bin/bash
module load conda/${BASE_CONDA} && \
conda activate ${ENV_PATH}/rapids-${RAPIDS_VERSION}_${SYSTEM} && \
\$@
EOF
