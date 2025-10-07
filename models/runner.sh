#!/bin/bash
#SBATCH --job-name=lstm_hyperparam        # Job name
#SBATCH --output=lstm_hyperparam_%j.out   # Output file
#SBATCH --error=lstm_hyperparam_%j.err    # Error file
#SBATCH --exclude=a256-t4-[01-04]         # Exclude slower nodes
#SBATCH --gres=gpu:1                      # Request 1 GPU
#SBATCH --mem=32G                         # Memory per node
#SBATCH --time=6:00:00                    # Time limit hrs:min:sec

PY_SCRIPT=lstm_classifiers.py
CONTAINER="../containers/torch-container.sif"

singularity exec --nv $CONTAINER python $PY_SCRIPT --language "ko" --column "question"
echo "-----------------------------------------------"
echo "Finished Korean model training."
echo "-----------------------------------------------"

singularity exec --nv $CONTAINER python $PY_SCRIPT --language "ar" --column "question"
echo "-----------------------------------------------"
echo "Finished Arabic model training."
echo "-----------------------------------------------"

singularity exec --nv $CONTAINER python $PY_SCRIPT --language "te" --column "question"
echo "-----------------------------------------------"
echo "Finished Telugu model training."
echo "-----------------------------------------------"

singularity exec --nv $CONTAINER python $PY_SCRIPT --language "eng" --column "context"
echo "-----------------------------------------------"
echo "Finished English model training."
echo "-----------------------------------------------"

echo "All model trainings complete."