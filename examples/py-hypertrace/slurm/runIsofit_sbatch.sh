#!/bin/bash
#SBATCH --job-name=IsoFit-slurm
#SBATCH --out="IsoFit_slurm_job-%j.out"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=25
#SBATCH	--tasks-per-node=1
#SBATCH	--mem-per-cpu=4GB
#SBATCH --partition compute
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sserbin@bnl.gov

### Set script options:
workflow_config=myconfig.json
conda_env=/data2/sserbin/conda_envs/isofit_develop

echo " "
echo "Starting at: " `date`
echo "Job submitted to the ${SLURM_JOB_PARTITION} partition on ${SLURM_CLUSTER_NAME}"
echo "Job name: ${SLURM_JOB_NAME}, Job ID: ${SLURM_JOB_ID}"
echo "  I have ${SLURM_CPUS_ON_NODE} CPUs on compute node $(hostname)"

echo "Run started on: " `date`

### define any required modules, e.g. 
module load libradtran

### set conda environment, if required. e.g.
# https://github.com/conda/conda/issues/7980
# https://github.com/conda/conda/issues/8536
#
# to find base environment, run "conda info | grep -i 'base environment'"
# replace prefix $conda_base_env with path to conda location
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
python workflow.py ${workflow_config}
python summarize.py ${workflow_config}

echo " "
echo "Run completed on: " `date`
