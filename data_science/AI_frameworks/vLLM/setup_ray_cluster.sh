
########################################################################
# FUNCTIONS
########################################################################

# Setup environment and variables needed to setup ray and vllm
setup_environment() {
    echo "[$(hostname)] Setting up the environment..."
    # Set proxy configurations
    export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
    export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
    export http_proxy="http://proxy.alcf.anl.gov:3128"
    export https_proxy="http://proxy.alcf.anl.gov:3128"
    export ftp_proxy="http://proxy.alcf.anl.gov:3128"
    
    # Define the common setup script path (make sure this file is accessible on all nodes)
    export COMMON_SETUP_SCRIPT="/path/to/setup_ray_cluster.sh"
    
    # Load modules and activate your conda environment
    module load frameworks
    conda activate vllm
    
    module unload oneapi/eng-compiler/2024.07.30.002
    module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
    module use /soft/preview/pe/24.347.0-RC2/modulefiles
    module add oneapi/release

    export TORCH_LLM_ALLREDUCE=1
    export CCL_ZE_IPC_EXCHANGE=drmfd

    export ZE_FLAT_DEVICE_HIERARCHY=FLAT

    export HF_TOKEN="YOUR_HF_TOKEN"
    export HF_HOME="/flare/datasets/model-weights/hub"
    export HF_DATASETS_CACHE="/flare/datasets/model-weights/hub"
    export TMPDIR="/tmp"

    export RAY_TMPDIR="/tmp"
    export VLLM_IMAGE_FETCH_TIMEOUT=60

    ulimit -c unlimited

    # Derive the node's HSN IP address (modify the getent command as needed)
    export HSN_IP_ADDRESS=$(getent hosts "$(hostname).hsn.cm.aurora.alcf.anl.gov" | awk '{ print $1 }' | sort | head -n 1)
    export VLLM_HOST_IP="$HSN_IP_ADDRESS"

    echo "[$(hostname)] Environment setup complete. HSN_IP_ADDRESS is $HSN_IP_ADDRESS"
}

# Stop any running Ray processes
stop_ray() {
    echo "[$(hostname)] Stopping Ray (if running)..."
    ray stop -f
}

# Start Ray head node
start_ray_head() {
    echo "[$(hostname)] Starting Ray head..."
    ray start --num-gpus=8 --num-cpus=64 --head --node-ip-address="$HSN_IP_ADDRESS" --temp-dir=/tmp

    # Wait until Ray reports that the head node is up
    echo "[$(hostname)] Waiting for Ray head to be up..."
    until ray status &>/dev/null; do
        sleep 5
        echo "[$(hostname)] Waiting for Ray head..."
    done
    echo "[$(hostname)] ray status: $(ray status)"
    echo "[$(hostname)] Ray head node is up."
}

# Start Ray worker node
start_ray_worker() {
    echo "[$(hostname)] Starting Ray worker, connecting to head at $RAY_HEAD_IP..."
    echo "HSN IP Address : $HSN_IP_ADDRESS"
    ray start --num-gpus=8 --num-cpus=64 --address="$RAY_HEAD_IP:6379" --node-ip-address="$HSN_IP_ADDRESS" --temp-dir=/tmp

    echo "[$(hostname)] Waiting for Ray worker to be up..."
    until ray status &>/dev/null; do
        sleep 5
        echo "[$(hostname)] Waiting for Ray worker..."
    done
    echo "[$(hostname)] ray status: $(ray status)"
    echo "[$(hostname)] Ray worker node is up."
}

########################################################################
# MAIN SCRIPT LOGIC
########################################################################


main() {

    # Ensure that the script is being run within a PBS job
    if [ -z "$PBS_NODEFILE" ]; then
        echo "Error: PBS_NODEFILE not set. This script must be run within a PBS job allocation."
        exit 1
    fi

    # Read all nodes from the PBS_NODEFILE into an array.
    mapfile -t nodes_full < "$PBS_NODEFILE"
    num_nodes=${#nodes_full[@]}

    echo "Allocated nodes ($num_nodes):"
    printf " - %s\n" "${nodes_full[@]}"

    # Require at least 2 nodes (one head + one worker)
    if [ "$num_nodes" -lt 2 ]; then
        echo "Error: Need at least 2 nodes to launch the Ray cluster."
        exit 1
    fi

    # The first node will be our Ray head.
    head_node_full="${nodes_full[0]}"

    # All remaining nodes will be the workers.
    worker_nodes_full=("${nodes_full[@]:1}")

    # It is a good idea to run this master script on the designated head node.
    current_node=$(hostname -f)


    echo "[$(hostname)] Running on head node."

    # --- Setup and start the head node ---
    setup_environment
    stop_ray
    start_ray_head

    # Export the head node's IP so that workers can join.
    export RAY_HEAD_IP="$HSN_IP_ADDRESS"
    echo "[$(hostname)] RAY_HEAD_IP exported as $RAY_HEAD_IP"

    # --- Launch Ray workers on each of the other nodes via SSH ---
    for worker in "${worker_nodes_full[@]}"; do
        echo "[$(hostname)] Launching Ray worker on $worker..."
        ssh "$worker" "bash -l -c 'set -x; export RAY_HEAD_IP=${RAY_HEAD_IP}; export COMMON_SETUP_SCRIPT="/path/to/setup_ray_cluster.sh" ;source \$COMMON_SETUP_SCRIPT; setup_environment; stop_ray; start_ray_worker'" &
    done

    # Wait for all background SSH jobs to finish.
    wait

    echo "[$(hostname)] Ray cluster is up and running with $num_nodes nodes."
}


