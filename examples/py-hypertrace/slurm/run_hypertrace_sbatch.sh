#!/bin/bash
#######################################################################################
#SBATCH --job-name=hypertrace-slurm
#SBATCH --out="hypertrace_job-%j.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH	--tasks-per-node=1
#SBATCH	--mem-per-cpu=4GB
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
### SLURM usage example
# sbatch -w node03 -c 12 --partition compute --job-name=py-hypertrace --mail-user=sserbin@bnl.gov slurm/run_hypertrace_sbatch.sh configs/libradtran.json
#######################################################################################

#######################################################################################
echo " "
echo "Starting at: " `date`
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "There are ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"

echo "Run started on: " `date`

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

### run workflow and summary script
echo " "
echo "Run Hypertrace workflow"
python workflow.py ${1}
python summarize.py ${1}

echo " "
echo "Run completed on: " `date`
#######################################################################################

### EOF