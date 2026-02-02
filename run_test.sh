#!/bin/bash

set -e

# Change to the script's directory to ensure relative paths work correctly
cd "$(dirname "$0")"

DATA_DIR="build/data/sift"
APPS_DIR="build/apps"
UTILS_DIR="${APPS_DIR}/utils"

# Ensure data directory exists
mkdir -p ${DATA_DIR}

echo "--- 0. Cleaning up old files ---"
INDEX_PREFIX="${DATA_DIR}/disk_index_sift"
RESULT_PATH="${DATA_DIR}/res"
GT_FILE="${DATA_DIR}/sift_base_query_gt100"

rm -f ${INDEX_PREFIX}_*
rm -f ${RESULT_PATH}_*
rm -f ${GT_FILE}
echo "Cleanup complete."

echo "--- 1. Converting data to binary format (if needed) ---"
# Check if bin files exist, if not convert
if [ ! -f "${DATA_DIR}/sift_learn.bin" ]; then
    echo "Converting sift_learn.fvecs to bin..."
    ${UTILS_DIR}/fvecs_to_bin float ${DATA_DIR}/sift_learn.fvecs ${DATA_DIR}/sift_learn.bin
fi
if [ ! -f "${DATA_DIR}/sift_query.bin" ]; then
    echo "Converting sift_query.fvecs to bin..."
    ${UTILS_DIR}/fvecs_to_bin float ${DATA_DIR}/sift_query.fvecs ${DATA_DIR}/sift_query.bin
fi

echo "--- 2. Computing ground truth ---"
# Note: Computing GT for 100K points
echo "Computing ground truth to ${GT_FILE}..."
${UTILS_DIR}/compute_groundtruth --data_type float --dist_fn l2 \
    --base_file ${DATA_DIR}/sift_learn.bin \
    --query_file ${DATA_DIR}/sift_query.bin \
    --gt_file ${GT_FILE} \
    --K 100

echo "--- 3. Building Disk Index ---"
echo "Building index to ${INDEX_PREFIX}..."
# Using -B 0.003 (3MB) for search budget as per example (fits 100K vectors with 32-byte PQ)
${APPS_DIR}/build_disk_index --data_type float --dist_fn l2 \
    --data_path ${DATA_DIR}/sift_learn.bin \
    --index_path_prefix ${INDEX_PREFIX} \
    -R 32 \
    -L 50 \
    -B 0.003 \
    -M 1

echo "--- 4. Searching Disk Index ---"
echo "Searching index..."
# Increased num_nodes_to_cache to 10000 for larger dataset
${APPS_DIR}/search_disk_index --data_type float --dist_fn l2 \
    --index_path_prefix ${INDEX_PREFIX} \
    --query_file ${DATA_DIR}/sift_query.bin \
    --gt_file ${GT_FILE} \
    -K 10 \
    -L 10 20 30 40 50 100 \
    --result_path ${RESULT_PATH} \
    --num_nodes_to_cache 10000

echo "--- Test script finished successfully! ---"
