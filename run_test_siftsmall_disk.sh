#!/bin/bash

set -e

# Change to the script's directory to ensure relative paths work correctly
cd "$(dirname "$0")"

DATA_DIR="build/data/siftsmall"
APPS_DIR="build/apps"
UTILS_DIR="${APPS_DIR}/utils"

# Ensure data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: ${DATA_DIR} does not exist. Please ensure siftsmall data is present."
    exit 1
fi

echo "--- 0. Cleaning up old files ---"
INDEX_PREFIX="${DATA_DIR}/disk_index_siftsmall"
RESULT_PATH="${DATA_DIR}/res"
GT_FILE="${DATA_DIR}/siftsmall_base_query_gt100"

rm -f ${INDEX_PREFIX}_*
rm -f ${RESULT_PATH}_*
rm -f ${GT_FILE}
echo "Cleanup complete."

echo "--- 1. Converting data to binary format (if needed) ---"
# Check if bin files exist, if not convert
if [ ! -f "${DATA_DIR}/siftsmall_learn.bin" ]; then
    echo "Converting siftsmall_learn.fvecs to bin..."
    ${UTILS_DIR}/fvecs_to_bin float ${DATA_DIR}/siftsmall_learn.fvecs ${DATA_DIR}/siftsmall_learn.bin
fi
if [ ! -f "${DATA_DIR}/siftsmall_query.bin" ]; then
    echo "Converting siftsmall_query.fvecs to bin..."
    ${UTILS_DIR}/fvecs_to_bin float ${DATA_DIR}/siftsmall_query.fvecs ${DATA_DIR}/siftsmall_query.bin
fi

echo "--- 2. Computing ground truth ---"
echo "Computing ground truth to ${GT_FILE}..."
${UTILS_DIR}/compute_groundtruth --data_type float --dist_fn l2 \
    --base_file ${DATA_DIR}/siftsmall_learn.bin \
    --query_file ${DATA_DIR}/siftsmall_query.bin \
    --gt_file ${GT_FILE} \
    --K 100

echo "--- 3. Building Disk Index ---"
echo "Building index to ${INDEX_PREFIX}..."
# Using -B 0.003 (3MB) for search budget
# Using -M 0.001 (1MB) for build budget to force disk-based build logic on small dataset
${APPS_DIR}/build_disk_index --data_type float --dist_fn l2 \
    --data_path ${DATA_DIR}/siftsmall_learn.bin \
    --index_path_prefix ${INDEX_PREFIX} \
    -R 32 \
    -L 50 \
    -B 0.003 \
    -M 0.001

echo "--- 4. Searching Disk Index ---"
echo "Searching index..."
${APPS_DIR}/search_disk_index --data_type float --dist_fn l2 \
    --index_path_prefix ${INDEX_PREFIX} \
    --query_file ${DATA_DIR}/siftsmall_query.bin \
    --gt_file ${GT_FILE} \
    -K 10 \
    -L 10 20 30 40 50 100 \
    --result_path ${RESULT_PATH} \
    --num_nodes_to_cache 100

echo "--- Test script finished successfully! ---"
