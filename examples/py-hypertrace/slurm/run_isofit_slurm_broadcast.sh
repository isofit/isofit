#!/bin/bash
#SBATCH --job-name=isofit-broadcast
#SBATCH --out="isofit_broadcast-%j.out"
#SBATCH --nodes=2
#SBATCH --cpus-per-task=3
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --partition compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sserbin@bnl.gov

### Usage example:
# sbatch slurm/runIsofit_slurm_broadcast.sh myconfig.json

### Set script options:
module load libradtran
conda_base_env=/home/sserbin/miniconda3/
conda_env=/data2/sserbin/conda_envs/isofit_develop
# Must be one less that the total number of nodes
#worker_num=2

### Setup conda environment:
source ${conda_base_env}etc/profile.d/conda.sh
conda activate ${conda_env}

### Display PATH
echo "PATH: "${PATH}

### setup Ray environment:
echo "*Setup Ray environment*"
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
echo "*Availible nodes: "${nodes}
nodes_array=( $nodes )

# Must be one less that the total number of nodes
worker_num=$(( ${#nodes_array[@]} - 1 ))
node1=${nodes_array[0]}

echo "Starting on: "${nodes_array}" node"

# Start the main node, and get the redis_password
echo "*Start the main Ray node & set the redis password*"
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

#export ip_head

# Start the head
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password &
echo "Head started on $node1"

# Give the head time to start before connecting workers
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

# Start the worker
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "starting on node $node2"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus=$SLURM_CPUS_PER_TASK --block --address=$ip_head --redis-password=$redis_password &
  #srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

total_cores=$(( SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK ))
echo "Total Cores: ${total_cores}"

python slurm/set_ray_params.py ${1} $redis_password $ip_head $total_cores
#python slurm/set_ray_params.py ${workflow_config} $redis_password $ip_head $total_cores
#python -u -c slurm/trainer.py
#python python workflow.py ${workflow_config}
python workflow.py myconfig.json
