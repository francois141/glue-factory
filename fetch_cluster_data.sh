#!/bin/bash

echo "Transferring training outputs on the local computer"

# Replace with your ETHZ username
REMOTE_USER="fcosta"       
REMOTE_HOST="euler.ethz.ch"
REMOTE_PATH="/cluster/scratch/$REMOTE_USER/outputs/"
LOCAL_PATH="./outputs/"

rsync -avz --progress --include='*/' --include='*checkpoint_best.tar' --exclude='*' "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}" "${LOCAL_PATH}"

echo "Transferring the dataset - abort script if you don't need this part"

REMOTE_DATASET=oxparis_100k
REMOTE_PATH_DATASET="/cluster/scratch/$REMOTE_USER/outputs/results/$REMOTE_DATASET"
LOCAL_PATH_DATASET="./data/$REMOTE_DATASET/"

rsync -avz --progress "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH_DATASET}" "${LOCAL_PATH_DATASET}"