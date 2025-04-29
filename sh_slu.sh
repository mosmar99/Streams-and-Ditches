#!/bin/bash -l

#SBATCH --job-name=slu

# Resource Allocation

# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start. Here, we say the job will take 20 minutes
#                d-hh:mm:ss
#SBATCH --time=0-3:30:00
# Define resources to use for the defined job. Resources, which are not defined
# will not be provided.

# For simplicity, keep the number of tasks to one
#SBATCH --ntasks 1 
# Select number of required GPUs (maximum 1)
#SBATCH --gres=gpu:1
# Select number of required CPUs per task (maximum 16)
#SBATCH --cpus-per-task 4
# Select the partition - use the priority partition if you are in the user group slurmPrio
# If you are not in that group, your jobs won't get scheduled - so remove the entry below or change the partition name to 'scavenger'
# Note that your jobs may be interrupted and restarted when run on the scavenger partition
#SBATCH --partition priority
# If you schedule your jobs on the 'scavenger' partition and you want them to be requeued instead of cancelled, you need to remove the leading # sign
##SBATCH --requeue

# you may not place bash commands before the last SBATCH directive
echo "now processing task id:: ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"

# specify model
MODEL="m1"

# Create a main log directory if it doesn't exist
MAIN_LOG_DIR="logs/${MODEL}"
mkdir -p "$MAIN_LOG_DIR"

# Create a subdirectory for this job's logs
JOB_LOG_DIR="${MAIN_LOG_DIR}/ID_${SLURM_JOB_ID}"
mkdir "$JOB_LOG_DIR"

# Move the slurm output file to the job log directory
mv "slurm-${SLURM_JOB_ID}.out" "$JOB_LOG_DIR/slurm-${SLURM_JOB_ID}.out"

source ~/micromamba/etc/profile.d/micromamba.sh
micromamba activate dl

# Run your python script and redirect output to a file within the job's log directory
python ./src/models/${MODEL}.py "$JOB_LOG_DIR" --epochs 50 > "${JOB_LOG_DIR}/output_${SLURM_JOB_ID}.txt"

echo "finished task with id:: ${SLURM_JOB_ID}"
# happy end
exit 0