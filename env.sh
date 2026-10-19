# source this file: puts every stage folder on PYTHONPATH and points the pipeline at its data root.
#   source env.sh            (Git Bash / Linux)
export PIPELINE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$PIPELINE_ROOT:$PIPELINE_ROOT/formfinding:$PIPELINE_ROOT/vares:$PIPELINE_ROOT/abaqus:$PIPELINE_ROOT/figures${PYTHONPATH:+:$PYTHONPATH}"
export PIPELINE_DATA="${PIPELINE_DATA:-$PIPELINE_ROOT/data}"
echo "PIPELINE_ROOT=$PIPELINE_ROOT  PIPELINE_DATA=$PIPELINE_DATA"
