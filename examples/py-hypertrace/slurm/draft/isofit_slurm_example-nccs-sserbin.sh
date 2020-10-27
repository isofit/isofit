#!/bin/bash

#SBATCH --job-name=isofit-broadcast
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4GB
#SBATCH --nodes=5
#SBATCH --tasks-per-node=1

# load required modules and conda environments
module load anaconda3
conda activate /att/gpfsfs/briskfs01/ppl/sserbin/conda_envs/isofit_wrapper

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

# Must be one less that the total number of nodes
worker_num=$(( ${#nodes_array[@]} - 1 ))

node1=${nodes_array[0]}

# Start the main node, and get the redis_password
ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
#redis_password=$(uuidgen)
redis_password=28956yjf90j239056u90#$t52

# Start the head
srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password &
echo "Head started on $node1"

# Give the head time to start before connecting workers
sleep 2

# Start the worker
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "starting on node $node2"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus=$SLURM_CPUS_PER_TASK --block --address=$ip_head --redis-password=$redis_password &
  # Flag --block will keep ray process alive on each compute node.
  sleep 2
done

total_cores=$(( SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK ))
echo "Total Cores: ${total_cores}"

python set_ray_params.py ${1} $redis_password $ip_head $total_cores
#python -u -c "from isofit.core.isofit import Isofit; model = Isofit(\"${1}\", logfile=\"${2}\"); model.run()"
python workflow.py myconfig.json
