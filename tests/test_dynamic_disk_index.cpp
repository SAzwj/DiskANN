#include "dynamic_disk_index.h"
#include "utils.h"
#include "disk_utils.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>

template<typename T>
void test_merge_scenario(const std::string& data_path, const std::string& index_prefix, const std::string& query_path) {
    diskann::IndexSearchParams search_params(20, 1);
    diskann::IndexWriteParametersBuilder write_params_builder(20, 32); 
    auto write_params = write_params_builder.build();

    // Use a small threshold to trigger merge quickly
    size_t mem_threshold = 50; 

    diskann::IndexConfigBuilder config_builder;
    auto config = config_builder
        .with_metric(diskann::Metric::L2)
        .with_dimension(128) 
        .with_max_points(mem_threshold * 2) 
        .with_data_type("float")
        .is_dynamic_index(true)
        .is_enable_tags(true)
        .is_concurrent_consolidate(true)
        .with_index_search_params(search_params)
        .with_index_write_params(write_params)
        .build();

    // Create a working copy of data
    std::string working_data_path = data_path + ".working_copy";
    {
        std::ifstream src(data_path, std::ios::binary);
        std::ofstream dst(working_data_path, std::ios::binary);
        dst << src.rdbuf();
    }
    std::cout << "Created working copy at: " << working_data_path << std::endl;

    // Clean up previous index files
    std::string cmd = "rm -f " + index_prefix + "*";
    system(cmd.c_str());

    // NOTE: In a real scenario, we might need an initial index. 
    // DynamicDiskIndex constructor calls load_disk_index. If it fails, it assumes empty.
    // But we need a valid initial disk index for the base data if we want to search it.
    // For this test, let's assume we start with the base data indexed.
    // However, DynamicDiskIndex currently doesn't build the initial index from data_path automatically if it's missing.
    // We should build it first.
    
    std::cout << "Building initial disk index..." << std::endl;
    std::string params = "32 50 0.003 0.001 " + std::to_string(std::thread::hardware_concurrency());
    diskann::build_disk_index<T>(working_data_path.c_str(), index_prefix.c_str(), params.c_str(), diskann::Metric::L2);
    std::cout << "Initial index built." << std::endl;

    std::cout << "Initializing DynamicDiskIndex..." << std::endl;
    diskann::DynamicDiskIndex<T> dynamic_index(config, working_data_path, index_prefix, mem_threshold);

    // Load queries to use as inserted data
    T* queries = nullptr;
    size_t num_queries, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_path, queries, num_queries, query_dim, query_aligned_dim);

    size_t k = 10;
    size_t l = 40; // Increased L to improve recall
    std::vector<uint64_t> indices(k);
    std::vector<float> distances(k);

    // Insert points to trigger merge
    size_t num_inserts = mem_threshold + 10; // Trigger merge
    size_t start_label = 1000000;
    std::vector<uint32_t> inserted_labels;

    std::cout << "Inserting " << num_inserts << " points..." << std::endl;
    for (size_t i = 0; i < num_inserts; ++i) {
        size_t query_idx = i % num_queries;
        uint32_t label = start_label + i;
        dynamic_index.insert(queries + query_idx * query_aligned_dim, label);
        inserted_labels.push_back(label);
    }
    std::cout << "Insertion complete. Merge should have been triggered." << std::endl;

    // Verify insertion after merge
    std::cout << "Verifying inserted points..." << std::endl;
    size_t found_count = 0;
    for (size_t i = 0; i < num_inserts; ++i) {
        size_t query_idx = i % num_queries;
        uint32_t expected_label = inserted_labels[i];

        dynamic_index.search(queries + query_idx * query_aligned_dim, k, l, indices.data(), distances.data());

        bool found = false;
        for (size_t j = 0; j < k; ++j) {
            if (indices[j] == expected_label) {
                found = true;
                break;
            }
        }
        if (found) {
            found_count++;
        } else {
            std::cout << "Point " << expected_label << " not found." << std::endl;
        }
    }

    std::cout << "Found " << found_count << " / " << num_inserts << " inserted points." << std::endl;

    if (found_count == num_inserts) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }

    diskann::aligned_free(queries);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <data_bin> <index_prefix> <query_bin>" << std::endl;
        return 1;
    }

    std::string data_path = argv[1];
    std::string index_prefix = argv[2];
    std::string query_path = argv[3];

    try {
        test_merge_scenario<float>(data_path, index_prefix, query_path);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
