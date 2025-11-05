#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 1. Clean up old files ---
echo "--- Cleaning up old files and directories ---"
rm -rf build/data/sift
rm -rf build/data/siftsmall
mkdir -p build/data/sift
echo "--- Cleanup complete ---"
echo

# --- 2. Generate synthetic data ---
echo "--- Generating 16-dimensional synthetic data (20,000 base points, 100 query points) ---"
# Generate the base dataset for building the index: 20,000 points, 16 dimensions
./build/apps/utils/generate_test_data build/data/sift/sift_learn.bin 5000 16
# Generate the query dataset for searching the index: 100 points, 16 dimensions
./build/apps/utils/generate_test_data build/data/sift/sift_query.bin 100 16
echo "--- Data generation complete ---"
echo

# --- 3. Compute ground truth ---
echo "--- Computing ground truth (K=100) ---"
./build/apps/utils/compute_groundtruth --data_type float --dist_fn l2 --base_file build/data/sift/sift_learn.bin --query_file build/data/sift/sift_query.bin --gt_file build/data/sift/sift_query_learn_gt100 --K 100
echo "--- Ground truth computation complete ---"
echo

# --- 4. Build the disk index ---
echo "--- Building disk index (R=32, L=50) ---"
./build/apps/build_disk_index --data_type float --dist_fn l2 --data_path build/data/sift/sift_learn.bin --index_path_prefix build/data/sift/disk_index_sift_learn_R32_L50_A1.2 -R 32 -L 50 -B 0.003 -M 0.001
echo "--- Index build complete ---"
echo

# --- 5. Search the index and measure recall ---
echo "--- Searching index and measuring recall (K=10, L=10,20,30,40,50,100) ---"
./build/apps/search_disk_index --data_type float --dist_fn l2 --index_path_prefix build/data/sift/disk_index_sift_learn_R32_L50_A1.2 --query_file build/data/sift/sift_query.bin --gt_file build/data/sift/sift_query_learn_gt100 -K 10 -L 10 20 30 40 50 100 --result_path build/data/sift/res --num_nodes_to_cache 1000
echo "--- Search and recall measurement complete ---"
echo

echo "--- Test script finished successfully! ---"
