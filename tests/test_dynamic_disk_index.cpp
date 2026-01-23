#include "dynamic_disk_index.h"
#include "utils.h"
#include "disk_utils.h"
#include <iostream>
#include <vector>
#include <cassert>

template<typename T>
void test_dynamic_disk_index(const std::string& data_path, const std::string& index_prefix, const std::string& query_path, const std::string& gt_path) {
    diskann::IndexSearchParams search_params(20, 1);
    diskann::IndexWriteParametersBuilder write_params_builder(20, 32); // L=20, R=32
    auto write_params = write_params_builder.build();

    diskann::IndexConfigBuilder config_builder;
    auto config = config_builder
        .with_metric(diskann::Metric::L2)
        .with_dimension(128) // siftsmall dimension
        .with_max_points(20000) // siftsmall size + buffer
        .with_data_type("float")
        .is_dynamic_index(true)
        .is_enable_tags(true)
        .is_concurrent_consolidate(true)
        .with_index_search_params(search_params)
        .with_index_write_params(write_params)
        .build();

    // Threshold for mem index merge
    size_t mem_threshold = 1000;

    std::cout << "Initializing DynamicDiskIndex..." << std::endl;
    diskann::DynamicDiskIndex<T> dynamic_index(config, index_prefix, mem_threshold);

    // Load queries
    T* queries = nullptr;
    size_t num_queries, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_path, queries, num_queries, query_dim, query_aligned_dim);

    // Search parameters
    size_t k = 10;
    size_t l = 20;
    std::vector<uint64_t> indices(k);
    std::vector<float> distances(k);

    std::cout << "Searching for first query..." << std::endl;
    dynamic_index.search(queries, k, l, indices.data(), distances.data());

    std::cout << "Results for query 0:" << std::endl;
    for (size_t i = 0; i < k; ++i) {
        std::cout << "  Rank " << i << ": ID " << indices[i] << ", Dist " << distances[i] << std::endl;
    }

    // Test Insertion
    std::cout << "\nTesting Insertion..." << std::endl;
    // Use the first query as a new point to insert
    // Use a label that is likely not in the dataset (e.g. 100000)
    uint32_t new_label = 100000;
    dynamic_index.insert(queries, new_label);

    // Search again, expecting the new point to be the nearest neighbor (dist 0)
    dynamic_index.search(queries, k, l, indices.data(), distances.data());
    
    bool found = false;
    for (size_t i = 0; i < k; ++i) {
        if (indices[i] == new_label) {
            found = true;
            std::cout << "  Found inserted point! ID " << indices[i] << ", Dist " << distances[i] << std::endl;
            if (std::abs(distances[i]) > 1e-5) {
                 std::cerr << "  Warning: Distance is not zero for self-search." << std::endl;
            }
            break;
        }
    }
    
    if (!found) {
        std::cerr << "Error: Inserted point not found in search results." << std::endl;
    }

    // Test Deletion
    std::cout << "\nTesting Deletion..." << std::endl;
    dynamic_index.remove(new_label);

    // Search again, expecting the point to be gone
    dynamic_index.search(queries, k, l, indices.data(), distances.data());
    
    found = false;
    for (size_t i = 0; i < k; ++i) {
        if (indices[i] == new_label) {
            found = true;
            break;
        }
    }

    if (found) {
        std::cerr << "Error: Deleted point still found in search results." << std::endl;
    } else {
        std::cout << "  Deleted point successfully filtered." << std::endl;
    }

    // Test Deletion from Disk Index
    // Pick a point that we know is in the disk index.
    // From previous search results, let's pick indices[0] (if it's not the inserted one)
    uint64_t disk_point_label = indices[0];
    if (disk_point_label == new_label) disk_point_label = indices[1]; // Should not happen if deleted

    std::cout << "\nTesting Deletion from Disk Index (Label " << disk_point_label << ")..." << std::endl;
    dynamic_index.remove((uint32_t)disk_point_label);

    dynamic_index.search(queries, k, l, indices.data(), distances.data());
    
    found = false;
    for (size_t i = 0; i < k; ++i) {
        if (indices[i] == disk_point_label) {
            found = true;
            break;
        }
    }

    if (found) {
        std::cerr << "Error: Disk point still found after deletion." << std::endl;
    } else {
        std::cout << "  Disk point successfully filtered." << std::endl;
    }

    diskann::aligned_free(queries);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <data_bin> <index_prefix> <query_bin> <gt_file>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_prefix = argv[2];
    std::string query_path = argv[3];
    std::string gt_path = argv[4];

    try {
        test_dynamic_disk_index<float>(data_path, index_prefix, query_path, gt_path);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
