#!/bin/bash -l

# Create a main log directory if it doesn't exist
MODEL="m1"
MAIN_LOG_DIR="logs/${MODEL}"
TIMESTAMP=$(date +%Y%m%d_%H%M)
mkdir -p "$MAIN_LOG_DIR"

# Create a subdirectory for this job's logs
JOB_LOG_DIR="${MAIN_LOG_DIR}/${TIMESTAMP}"
mkdir "$JOB_LOG_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl

echo "started task with timestamp :: ${TIMESTAMP}"

# Run your python script and redirect output to a file within the job's log directory
START=$(date +%s)
python main.py "$JOB_LOG_DIR" --epochs 1 > "${JOB_LOG_DIR}/output_${TIMESTAMP}.txt"
END=$(date +%s)
RUNTIME=$((END - START))


TIMESTAMP2=$(date +%Y%m%d_%H%M)
echo "finished task with timestamp id :: ${TIMESTAMP}, finished on ${TIMESTAMP2}"

# Calculate and log Days, Hours, Minutes and Seconds from runtime
DAYS=$((RUNTIME / 86400))
HOURS=$(( (RUNTIME % 86400) / 3600 ))
MINUTES=$(( (RUNTIME % 3600) / 60 ))
SECONDS=$((RUNTIME % 60))
echo "Total excecution time: ${DAYS}d ${HOURS}h ${MINUTES}m ${SECONDS}s."

# happy end
exit 0