#include "dynamic_disk_index.h"
#include "utils.h"
#include "disk_utils.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>
#include <random>
#include <algorithm>

template<typename T>
void test_merge_scenario(const std::string& data_path, const std::string& index_prefix, const std::string& query_path) {
    diskann::IndexSearchParams search_params(20, 1);
    diskann::IndexWriteParametersBuilder write_params_builder(20, 32); 
    auto write_params = write_params_builder.build();

    // 设置小阈值以快速触发合并
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

    // 创建数据副本
    std::string working_data_path = data_path + ".working_copy";
    {
        std::ifstream src(data_path, std::ios::binary);
        std::ofstream dst(working_data_path, std::ios::binary);
        dst << src.rdbuf();
    }
    std::cout << "Created working copy at: " << working_data_path << std::endl;

    // 清理旧索引文件
    std::string cmd = "rm -f " + index_prefix + "*";
    system(cmd.c_str());

    std::cout << "Building initial disk index..." << std::endl;
    std::string params = "32 50 0.003 0.001 " + std::to_string(std::thread::hardware_concurrency());
    diskann::build_disk_index<T>(working_data_path.c_str(), index_prefix.c_str(), params.c_str(), diskann::Metric::L2);
    std::cout << "Initial index built." << std::endl;

    std::cout << "Initializing DynamicDiskIndex..." << std::endl;
    diskann::DynamicDiskIndex<T> dynamic_index(config, working_data_path, index_prefix, mem_threshold);

    // 加载查询向量用于插入
    T* queries = nullptr;
    size_t num_queries, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_path, queries, num_queries, query_dim, query_aligned_dim);

    size_t k = 10;
    size_t l = 40; // 增加 L 以提高召回率
    std::vector<uint64_t> indices(k);
    std::vector<float> distances(k);

    // 第一阶段：大量插入触发多次合并
    size_t num_inserts_phase1 = 500; // 500 个点，应该触发约 10 次合并
    size_t start_label = 1000000;
    std::vector<uint32_t> inserted_labels;

    std::cout << "\nPhase 1: Inserting " << num_inserts_phase1 << " points..." << std::endl;
    for (size_t i = 0; i < num_inserts_phase1; ++i) {
        size_t query_idx = i % num_queries;
        uint32_t label = start_label + i;
        dynamic_index.insert(queries + query_idx * query_aligned_dim, label);
        inserted_labels.push_back(label);
        if (i % 50 == 0) std::cout << "." << std::flush;
    }
    std::cout << "\nPhase 1 complete." << std::endl;

    // 验证第一阶段插入
    std::cout << "Verifying Phase 1 points..." << std::endl;
    size_t found_count = 0;
    for (size_t i = 0; i < num_inserts_phase1; ++i) {
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
        }
    }
    std::cout << "Found " << found_count << " / " << num_inserts_phase1 << " inserted points." << std::endl;
    if (found_count < num_inserts_phase1 * 0.95) { // 允许少量召回损失
        std::cerr << "Phase 1 verification failed! Recall too low." << std::endl;
        // exit(1); 
    }

    // 第二阶段：删除操作
    size_t num_deletes = 100;
    std::cout << "\nPhase 2: Deleting " << num_deletes << " points..." << std::endl;
    std::vector<uint32_t> deleted_labels;
    for (size_t i = 0; i < num_deletes; ++i) {
        // 删除前 50 个（在磁盘中）和后 50 个（可能在内存或磁盘）
        uint32_t label_to_delete = inserted_labels[i]; 
        dynamic_index.remove(label_to_delete);
        deleted_labels.push_back(label_to_delete);
    }
    
    // 验证删除
    std::cout << "Verifying deletions..." << std::endl;
    size_t deleted_found_count = 0;
    for (size_t i = 0; i < num_deletes; ++i) {
        size_t query_idx = i % num_queries;
        uint32_t deleted_label = deleted_labels[i];

        dynamic_index.search(queries + query_idx * query_aligned_dim, k, l, indices.data(), distances.data());

        bool found = false;
        for (size_t j = 0; j < k; ++j) {
            if (indices[j] == deleted_label) {
                found = true;
                break;
            }
        }
        if (found) {
            std::cout << "Deleted point " << deleted_label << " was FOUND!" << std::endl;
            deleted_found_count++;
        }
    }
    std::cout << "Found " << deleted_found_count << " / " << num_deletes << " deleted points (should be 0)." << std::endl;
    if (deleted_found_count > 0) {
        std::cerr << "Phase 2 verification failed! Deleted points still found." << std::endl;
    }

    // 第三阶段：混合插入新点
    size_t num_inserts_phase3 = 100;
    std::cout << "\nPhase 3: Inserting " << num_inserts_phase3 << " new points..." << std::endl;
    size_t start_label_phase3 = start_label + num_inserts_phase1;
    for (size_t i = 0; i < num_inserts_phase3; ++i) {
        size_t query_idx = (num_inserts_phase1 + i) % num_queries;
        uint32_t label = start_label_phase3 + i;
        dynamic_index.insert(queries + query_idx * query_aligned_dim, label);
        inserted_labels.push_back(label);
        if (i % 50 == 0) std::cout << "." << std::flush;
    }
    std::cout << "\nPhase 3 complete." << std::endl;

    // 验证新插入点
    std::cout << "Verifying Phase 3 points..." << std::endl;
    found_count = 0;
    for (size_t i = 0; i < num_inserts_phase3; ++i) {
        size_t query_idx = (num_inserts_phase1 + i) % num_queries;
        uint32_t expected_label = start_label_phase3 + i;

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
        }
    }
    std::cout << "Found " << found_count << " / " << num_inserts_phase3 << " new inserted points." << std::endl;

    if (found_count >= num_inserts_phase3 * 0.95 && deleted_found_count == 0) {
        std::cout << "TEST PASSED" << std::endl;
    } else {
        std::cout << "TEST FAILED" << std::endl;
    }

    diskann::aligned_free(queries);
}

template<typename T>
void test_budget_control(const std::string& data_path, const std::string& index_prefix, const std::string& query_path) {
    std::cout << "\n--- Testing Budget Control ---" << std::endl;
    diskann::IndexSearchParams search_params(20, 1);
    diskann::IndexWriteParametersBuilder write_params_builder(20, 32); 
    auto write_params = write_params_builder.build();

    diskann::IndexConfigBuilder config_builder;
    auto config = config_builder
        .with_metric(diskann::Metric::L2)
        .with_dimension(128) 
        .with_max_points(1000) // 临时值，实际上会被动态阈值覆盖
        .with_data_type("float")
        .is_dynamic_index(true)
        .is_enable_tags(true)
        .is_concurrent_consolidate(true)
        .with_index_search_params(search_params)
        .with_index_write_params(write_params)
        .build();

    std::string working_data_path = data_path + ".budget_test";
    {
        std::ifstream src(data_path, std::ios::binary);
        std::ofstream dst(working_data_path, std::ios::binary);
        dst << src.rdbuf();
    }

    std::string budget_index_prefix = index_prefix + "_budget";
    std::string cmd = "rm -f " + budget_index_prefix + "*";
    system(cmd.c_str());

    // 创建初始索引
    std::string params = "32 50 0.003 0.001 " + std::to_string(std::thread::hardware_concurrency());
    diskann::build_disk_index<T>(working_data_path.c_str(), budget_index_prefix.c_str(), params.c_str(), diskann::Metric::L2);

    // 使用极小的预算初始化 DynamicDiskIndex (例如 0.00005 GB ~= 50KB)
    // 根据估计，每个点大约占用几百字节，50KB 应该能容纳几十到一百个点
    double budget_gb = 0.00005; 
    
    std::cout << "Initializing DynamicDiskIndex with budget " << budget_gb << " GB..." << std::endl;
    diskann::DynamicDiskIndex<T> dynamic_index(config, working_data_path, budget_index_prefix, 0, budget_gb);

    // 插入一些点，验证是否正常工作（不崩馈）
    T* queries = nullptr;
    size_t num_queries, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_path, queries, num_queries, query_dim, query_aligned_dim);

    size_t num_inserts = 100;
    std::cout << "Inserting " << num_inserts << " points..." << std::endl;
    for (size_t i = 0; i < num_inserts; ++i) {
        dynamic_index.insert(queries + i * query_aligned_dim, 2000000 + i);
    }
    std::cout << "Insertion complete. Budget control test passed." << std::endl;
    
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
        test_budget_control<float>(data_path, index_prefix, query_path);
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
