#!/bin/bash

#SBATCH --job-name=slurm_test
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=1GB
#SBATCH --nodes=3
#SBATCH --tasks-per-node 1

worker_num=2 # Must be one less that the total number of nodes

# module load Langs/Python/3.6.4 # This will vary depending on your environment
# source venv/bin/activate

echo '*** Load Modules ***'
#module load anaconda3

echo '*** Load Conda Env ***'
conda_base_env=/home/sserbin/miniconda3/
# for bash
source ${conda_base_env}etc/profile.d/conda.sh
#conda activate /att/gpfsfs/briskfs01/ppl/sserbin/conda_envs/isofit_wrapper
conda activate /data2/sserbin/conda_envs/isofit_develop

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)
#redis_password=1y689fni913h5#$nt5m2

export ip_head # Exporting for latter access by trainer.py

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 5
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 5
done

python -u trainer.py $redis_password 15 # Pass the total number of allocated CPUs

### EOF
