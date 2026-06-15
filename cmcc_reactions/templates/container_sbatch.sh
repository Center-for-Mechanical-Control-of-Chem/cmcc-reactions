#!/bin/bash
##NECESSARY JOB SPECIFICATIONS
#SBATCH --time=24:00:00 #Set the wall clock limit
#SBATCH --ntasks-per-node=1                   #Request tasks
#SBATCH --nodes=1                   #Request tasks
#SBATCH --mem=15G                  #Request Memory in MB per node
#SBATCH --output=%x-%j.out      #Send stdout/err

if [ -z "$CONTAINER_RUN_SCRIPT" ]; then
  CONTAINER_RUN_SCRIPT=`./CONTAINER_SCRIPT_NAME`
fi
if [ -z "$CONTAINER_PYTHON_PATH" ]; then
  CONTAINER_PYTHON_PATH="/home"
fi
export CONTAINER_PYTHON_PATH="$CONTAINER_PYTHON_PATH"
# bind in paths via --bind, pass as `sbatch --env=CONTAINER_ARGS="--bind=/path/to/dep:/home/dep,/path/to/other:/home/other"
if [ -z "$CONTAINER_ARGS" ]; then
  if [ -n "$CONTAINER_DEPENDENCY_PATH" ]; then
    CONTAINER_ARGS="--bind=$CONTAINER_DEPENDENCY_PATH/McUtils:$CONTAINER_PYTHON_PATH/McUtils,$CONTAINER_DEPENDENCY_PATH/Psience:$CONTAINER_PYTHON_PATH/Psience,$CONTAINER_DEPENDENCY_PATH/cmcc_reactions:$CONTAINER_PYTHON_PATH/cmcc_reactions"
  fi
fi
export CONTAINER_ARGS="$CONTAINER_ARGS"
if [ -z "$CONTAINER_ENV" ]; then
  CONTAINER_ENV="cmcc-rxn"
fi
if [ -z "$CONTAINER_COMMAND" ]; then
  CONTAINER_COMMAND="python -u"
fi
if [ -z "$INPUT_FILE" ]; then
  if [ -z "$IGNORE_INPUT_FILE" ]; then
    if [ "$SLURM_JOB_NAME" != "${SLURM_JOB_NAME%.*}" ];
      then
        INPUT_FILE="$SLURM_JOB_NAME"
      else
        INPUT_FILE="${SLURM_JOB_NAME%.*}.py"
    fi
  fi
fi

echo "Starting Job $SLURM_JOB_NAME"
START=$(date +%s.%N)
echo "  START: $(date)"
echo "    PWD: $PWD"
echo "  NODES: $SLURM_JOB_NUM_NODES"
echo "  PART.: $SLURM_JOB_PARTITION"
echo "=================================================="

$CONTAINER_RUN_SCRIPT --env=$CONTAINER_ENV $CONTAINER_COMMAND $INPUT_FILE $@

echo "=================================================="
END=$(date +%s.%N)
DIFF=$(awk "BEGIN {print $END - $START}")
echo "   END: $(date)"
echo "  TIME: $DIFF"

# SCRATCH=/scratch/user/`whoami`

