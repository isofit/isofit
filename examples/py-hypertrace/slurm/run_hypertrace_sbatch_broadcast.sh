#!/bin/bash
#######################################################################################
#SBATCH --job-name=hypertrace-slurm
#SBATCH --out="hypertrace_job-%j.out"
#SBATCH --nodes=2
#SBATCH --cpus-per-task=3
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --mail-type=ALL
#######################################################################################

#######################################################################################
### Set script options:
#conda_env=/data2/sserbin/conda_envs/isofit_develop
conda_env=/data2/sserbin/conda_envs/isofit_develop

# define any required modules, e.g. 
#module load libradtran gdal/3.1.2_hdf4-gcc840 proj/7.1.0-gcc840 geos/3.8.1-gcc840 
module load libradtran gdal/3.1.2_hdf4-gcc840 proj/7.1.0-gcc840 geos/3.8.1-gcc840 
#######################################################################################

#######################################################################################
### Usage example:
# sbatch --partition compute --job-name=py-hypertrace --mail-user=sserbin@bnl.gov slurm/run_hypertrace_sbatch_broadcast.sh configs/libradtran.json
#######################################################################################

#######################################################################################
echo " "
echo "Starting at: " `date`

### set conda environment, if required. e.g.
# https://github.com/conda/conda/issues/7980
# https://github.com/conda/conda/issues/8536
#
# to find base environment, run "conda info | grep -i 'base environment'"
# replace prefix $conda_base_env with path to conda location
# e.g. conda_base_env=/home/sserbin/miniconda3/
conda_base_env=/home/sserbin/miniconda3/
# for bash
source ${conda_base_env}etc/profile.d/conda.sh
#
# for csh
# source ${conda_base_env}etc/profile.d/conda.csh
# others may also be availible in ${conda_base_env}etc/profile.d/

# load conda environment
conda activate ${conda_env}
python --version
which python

### Display PATH
echo "PATH: "${PATH}
#######################################################################################

#######################################################################################
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
echo "Redis password $redis_password"
#######################################################################################

#######################################################################################
# Stop anything previous
for ((  i=0; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 rm -rf /tmp/ray/*
  srun --nodes=1 --ntasks=1 -w $node2 ray stop --force
  srun --nodes=1 --ntasks=1 -w $node2 rm -rf /dev/shm/*
done
#######################################################################################

#######################################################################################
# Start the head
srun --nodes=1 --ntasks=1 -w $node1 ray start --node-ip-address=$ip_prefix --num-cpus=$SLURM_CPUS_PER_TASK --block --head --port=6379 --redis-password=$redis_password &
echo "Head started on $node1"

# Give the head time to start before connecting workers
sleep 2
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.
#######################################################################################

#######################################################################################
# Start the worker
for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  echo "starting on node $node2"
  srun --nodes=1 --ntasks=1 -w $node2 ray start --num-cpus=$SLURM_CPUS_PER_TASK --block --address=$ip_head --redis-password=$redis_password &
  # Flag --block will keep ray process alive on each compute node.
  sleep 1
done

total_cores=$(( SLURM_JOB_NUM_NODES * SLURM_CPUS_PER_TASK ))
echo "Total Cores: ${total_cores}"
#######################################################################################

#######################################################################################
### run workflow and summary script
echo " "
echo "Run Hypertrace workflow"
python slurm/set_ray_params.py ${1} $redis_password $ip_head $total_cores

python workflow.py ${1}
python summarize.py ${1}

echo " "
echo "Run completed on: " `date`
#######################################################################################

### EOF