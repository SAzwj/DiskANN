#include "dynamic_disk_index.h"
#include "common_includes.h"
#include "disk_utils.h"
#include "utils.h"
#include "linux_aligned_file_reader.h"
#include <algorithm>
#include <future>
#include <thread>

namespace diskann {

template<typename T, typename LabelT>
DynamicDiskIndex<T, LabelT>::DynamicDiskIndex(const IndexConfig& config, const std::string& data_file_path, const std::string& disk_index_path, size_t mem_index_threshold)
    : _config(config), _data_file_path(data_file_path), _disk_index_path(disk_index_path), _mem_index_threshold(mem_index_threshold) {
    
    // 设置内存索引的最大容量
    size_t max_points = _mem_index_threshold * 2;
    
    auto write_params = config.index_write_params;
    auto search_params = config.index_search_params;

    // 初始化内存索引（Vamana 索引）
    _mem_index = std::make_shared<Index<T, LabelT>>(
        Metric::L2,
        config.dimension,
        max_points,
        write_params,
        search_params,
        0,     // num_frozen_pts: 冻结点数
        true,  // dynamic_index: 启用动态索引
        true,  // enable_tags: 启用标签
        true,  // concurrent_consolidate: 启用并发整合
        false, // pq_dist_build: 禁用 PQ 距离构建
        0,     // num_pq_chunks: PQ 块数
        false, // use_opq: 禁用 OPQ
        false  // filtered_index: 禁用过滤索引
    );

    _mem_index->init_empty_index();

    // 加载现有的磁盘索引
    load_disk_index();
}

template<typename T, typename LabelT>
DynamicDiskIndex<T, LabelT>::~DynamicDiskIndex() {
}

template<typename T, typename LabelT>
void DynamicDiskIndex<T, LabelT>::load_disk_index() {
    // 使用 Linux 异步对齐读取器加载磁盘上的 PQFlashIndex
    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();
    _disk_index = std::make_shared<PQFlashIndex<T, LabelT>>(reader, Metric::L2);

    int res = _disk_index->load(1, _disk_index_path.c_str());
    if (res != 0) {
        // 第一次运行时可能没有磁盘索引，这是正常的，可以忽略
        // 或者我们可以抛出异常，取决于设计。在这里我们假设如果是第一次，_disk_index 将为空或不可用
        // 但为了 DynamicDiskIndex 的正确性，我们可能期望它至少能处理空状态
        // 实际上，如果 load 失败，我们应该重置 _disk_index 或者标记为不可用
        // 这里为了简单，如果加载失败，我们假设没有磁盘索引
        _disk_index.reset();
        // throw ANNException("无法加载磁盘索引", -1, __FUNCSIG__, __FILE__, __LINE__);
        return;
    }

    // 构建磁盘索引标签到内部 ID 的映射
    size_t num_points = _disk_index->get_num_points();
    _disk_label_to_id.clear();
    for (uint32_t i = 0; i < num_points; ++i) {
        try {
            LabelT label = _disk_index->get_label(i);
            _disk_label_to_id[label] = i;
        } catch (...) {
            // 忽略读取标签失败的点
        }
    }
}

template<typename T, typename LabelT>
void DynamicDiskIndex<T, LabelT>::insert(const T* point, const LabelT label) {
    std::unique_lock<std::shared_mutex> lock(_rw_lock);
    
    // 处理覆盖插入逻辑：如果标签之前被标记为删除，则撤销删除
    if (_deleted_labels.find(label) != _deleted_labels.end()) {
        _deleted_labels.erase(label);
        
        // 如果该标签在磁盘索引中，从磁盘删除集合中移除
        if (_disk_label_to_id.find(label) != _disk_label_to_id.end()) {
            _disk_deleted_ids.erase(_disk_label_to_id[label]);
        }
    }

    std::vector<LabelT> labels = {label};
    int res = _mem_index->insert_point(point, label, labels);
    if (res != 0) {
        diskann::cerr << "Insert failed for label " << label << " with error code " << res << std::endl;
    }

    // 检查是否达到阈值触发合并逻辑
    bool trigger_merge = false;
    if (_mem_index->get_num_points() >= _mem_index_threshold) {
        trigger_merge = true;
    }
    lock.unlock();

    if (trigger_merge) {
        this->merge();
    }
}

template<typename T, typename LabelT>
void DynamicDiskIndex<T, LabelT>::remove(const LabelT label) {
    std::unique_lock<std::shared_mutex> lock(_rw_lock);
    
    // 记录全局删除标签
    _deleted_labels.insert(label);

    // 如果该标签存在于磁盘索引中，记录其对应的物理 ID
    auto it = _disk_label_to_id.find(label);
    if (it != _disk_label_to_id.end()) {
        _disk_deleted_ids.insert(it->second);
    }

    // 内存索引执行延迟删除
    // 只有当标签在内存索引中时才调用 lazy_delete，避免误报 "Delete tag not found"
    // 但是 Index 没有公开检查 tag 是否存在的接口，除了 search/get_vector_by_tag
    // 或者我们直接忽略 Index 内部打印的错误
    _mem_index->lazy_delete(label);
}

template<typename T, typename LabelT>
void DynamicDiskIndex<T, LabelT>::search(const T* query, size_t k, size_t l, uint64_t* indices, float* distances) {
    std::shared_lock<std::shared_mutex> lock(_rw_lock);

    // 同时检索内存索引和磁盘索引
    std::vector<LabelT> mem_indices(k * 2);
    std::vector<float> mem_distances(k * 2);
    std::vector<T*> res_vectors; 
    
    size_t mem_count = _mem_index->search_with_tags(query, k, l, mem_indices.data(), mem_distances.data(), res_vectors);
    // diskann::cout << "Mem search found " << mem_count << " results." << std::endl;

    std::vector<uint64_t> disk_indices_u64(k * 2);
    std::vector<float> disk_distances(k * 2);
    
    // 执行磁盘束搜索 (Beam Search)
    if (_disk_index) {
        _disk_index->cached_beam_search(query, k, l, disk_indices_u64.data(), disk_distances.data(), l, 
                                        false, 0, 
                                        std::numeric_limits<uint32_t>::max(), 
                                        false, 
                                        &_disk_deleted_ids);
    } else {
        // 如果磁盘索引未加载，初始化结果为最大距离
        std::fill(disk_distances.begin(), disk_distances.end(), std::numeric_limits<float>::max());
    }

    struct Result {
        LabelT label;
        float dist;
        bool operator<(const Result& other) const {
            return dist < other.dist;
        }
    };

    std::vector<Result> all_results;
    all_results.reserve(2 * k);

    // 过滤内存索引结果（排除已删除标签）
    size_t valid_mem_count = 0;
    for (size_t i = 0; i < mem_count; ++i) {
        if (_deleted_labels.find(mem_indices[i]) == _deleted_labels.end()) {
            all_results.push_back({mem_indices[i], mem_distances[i]});
            valid_mem_count++;
        }
    }

    if (valid_mem_count == 0 && mem_count > 0) {
        // diskann::cout << "All " << mem_count << " mem results filtered out." << std::endl;
    } else if (valid_mem_count == 0 && mem_count == 0) {
        // diskann::cout << "No mem results found." << std::endl;
    }

    // 过滤磁盘索引结果
    size_t valid_disk_count = 0;
    for (size_t i = 0; i < k; ++i) {
        if (disk_distances[i] == std::numeric_limits<float>::max()) continue;

        uint32_t id = (uint32_t)disk_indices_u64[i];
        
        // 检查磁盘 ID 是否已被逻辑删除
        if (_disk_deleted_ids.find(id) != _disk_deleted_ids.end()) {
            // diskann::cout << "Debug: Disk ID " << id << " is in _disk_deleted_ids" << std::endl;
            continue;
        }

        try {
            LabelT label = _disk_index->get_label(id);
            if (_deleted_labels.find(label) == _deleted_labels.end()) {
                all_results.push_back({label, disk_distances[i]});
            } else {
                // diskann::cout << "Debug: Label " << label << " (Disk ID " << id << ") is in _deleted_labels" << std::endl;
            }
        } catch (...) {
            continue;
        }
    }

    // 按距离排序并去重
    std::sort(all_results.begin(), all_results.end());
    
    std::vector<Result> unique_results;
    tsl::robin_set<LabelT> seen_labels;
    for (const auto& res : all_results) {
        if (seen_labels.find(res.label) == seen_labels.end()) {
            seen_labels.insert(res.label);
            unique_results.push_back(res);
            if (unique_results.size() >= k) break;
        }
    }

    // 填充结果数组
    for (size_t i = 0; i < k; ++i) {
        if (i < unique_results.size()) {
            indices[i] = (uint64_t)unique_results[i].label;
            distances[i] = unique_results[i].dist;
        } else {
            indices[i] = 0; 
            distances[i] = std::numeric_limits<float>::max();
        }
    }
    
    // DEBUG: Print search results
    // diskann::cout << "Search results (" << unique_results.size() << "): ";
    // for (const auto& res : unique_results) {
    //     diskann::cout << res.label << "(" << res.dist << ") ";
    // }
    // diskann::cout << std::endl;
}

template<typename T, typename LabelT>
void DynamicDiskIndex<T, LabelT>::merge() {
    std::unique_lock<std::shared_mutex> lock(_rw_lock);
    diskann::cout << "正在将内存索引合并至磁盘索引..." << std::endl;

    // 在压缩前整合内存索引的删除点
    _mem_index->consolidate_deletes(*_config.index_write_params);

    std::string temp_mem_index_path = _disk_index_path + "_temp_mem.index";
    std::string temp_mem_data_path = temp_mem_index_path + ".data";
    std::string temp_mem_tags_path = temp_mem_index_path + ".tags";

    // 保存内存索引的数据和标签
    _mem_index->save(temp_mem_index_path.c_str(), true);

    size_t num_active_points = _mem_index->get_num_points();

    // 加载保存的内存索引数据块
    size_t mem_num_points, mem_dim;
    std::unique_ptr<T[]> mem_data;
    diskann::load_bin<T>(temp_mem_data_path, mem_data, mem_num_points, mem_dim);

    std::unique_ptr<LabelT[]> mem_tags;
    size_t n_tags, dim_tags;
    diskann::load_bin<LabelT>(temp_mem_tags_path, mem_tags, n_tags, dim_tags);

    // 将内存中的数据追加到原始原始二进制数据文件中
    int32_t file_num_points_i32 = 0, file_dim_i32 = 0;
    bool is_new_file = false;
    
    // 检查文件是否存在和大小
    {
        std::ifstream data_reader(_data_file_path, std::ios::binary | std::ios::ate);
        size_t file_size = 0;
        if (data_reader.is_open()) {
            file_size = data_reader.tellg();
            data_reader.seekg(0, std::ios::beg);
        }

        if (file_size < 2 * sizeof(int32_t)) {
             std::cout << "Data file is new or empty. Size: " << file_size << std::endl;
             is_new_file = true;
             file_num_points_i32 = 0;
             file_dim_i32 = (int32_t)mem_dim;
        } else {
            data_reader.read((char*)&file_num_points_i32, sizeof(int32_t));
            data_reader.read((char*)&file_dim_i32, sizeof(int32_t));
        }
        if (data_reader.is_open()) data_reader.close();
    }

    size_t file_num_points = (size_t)file_num_points_i32;
    size_t file_dim = (size_t)file_dim_i32;

    if (file_dim != mem_dim) {
            std::cout << "Error: Dimension mismatch. File dim: " << file_dim << ", Mem dim: " << mem_dim 
                      << ", File num points: " << file_num_points << std::endl;
            if (file_dim == 0) {
                std::cout << "Warning: File dimension is 0, assuming correct dimension is " << mem_dim << std::endl;
                file_dim = mem_dim;
            } else {
                throw ANNException("合并期间维度不匹配", -1, __FUNCSIG__, __FILE__, __LINE__);
            }
    }

    {
        // 使用 fstream 进行读写
        std::fstream data_writer(_data_file_path, std::ios::binary | std::ios::in | std::ios::out);
        // 如果文件不存在或太小，可能需要以 trunc 模式打开一次来创建它，或者确保它至少有头部大小
        if (is_new_file) {
            data_writer.close(); // 先关闭可能打开失败的流
            // 以 trunc 模式创建文件
            data_writer.open(_data_file_path, std::ios::binary | std::ios::out | std::ios::trunc);
            // 写入头部占位
            int32_t zero = 0;
            int32_t dim_i32 = (int32_t)mem_dim;
            data_writer.write((char*)&zero, sizeof(int32_t));
            data_writer.write((char*)&dim_i32, sizeof(int32_t));
            // 重新以读写模式打开
            data_writer.close();
            data_writer.open(_data_file_path, std::ios::binary | std::ios::in | std::ios::out);
        }

        if (!data_writer.is_open()) {
             throw ANNException("无法打开数据文件进行写入", -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        // 定位到末尾并写入新数据
        data_writer.seekp(0, std::ios::end);
        size_t pos_before_write = data_writer.tellp();
        for (size_t i = 0; i < num_active_points; ++i) {
             data_writer.write((char*)(mem_data.get() + i * mem_dim), mem_dim * sizeof(T));
        }
        size_t pos_after_write = data_writer.tellp();

        // 更新数据文件头的总点数
        int32_t new_num_points_i32 = (int32_t)(file_num_points + num_active_points);
        data_writer.clear(); // 清除可能存在的 EOF 标志
        data_writer.seekp(0, std::ios::beg);
        data_writer.write((char*)&new_num_points_i32, sizeof(int32_t));
        // 如果不是新文件，也确保维度正确写入（如果是0的话）
        if (!is_new_file && file_dim_i32 == 0) {
            int32_t dim_i32 = (int32_t)mem_dim;
            data_writer.write((char*)&dim_i32, sizeof(int32_t));
        }
        
        data_writer.close();
    }

    // 设置磁盘索引构建参数 (R L B M T)
    std::string params = "32 50 0.003 0.001 " + std::to_string(std::thread::hardware_concurrency()); 
    std::string label_file_path = _disk_index_path + "_labels.txt";
    
    // 处理标签文件：确保标签文件与数据文件（合并前）的点数一致，然后追加新标签
    // initial_points 应该是追加新数据之前文件中的点数
    size_t initial_points = file_num_points;

    std::vector<LabelT> existing_labels;
    if (file_exists(label_file_path)) {
        std::ifstream label_reader(label_file_path);
        std::string line;
        while (std::getline(label_reader, line)) {
            try {
                if (!line.empty()) {
                    existing_labels.push_back((LabelT)std::stoul(line));
                }
            } catch (...) {
                break;
            }
        }
        label_reader.close();
    }

    // 如果读取的标签少于初始点数，补充默认标签（ID）
    if (existing_labels.size() < initial_points) {
        for (size_t i = existing_labels.size(); i < initial_points; ++i) {
            existing_labels.push_back((LabelT)i);
        }
    } else if (existing_labels.size() > initial_points) {
        // 如果标签多于数据点（可能是上次运行残留或错误），截断
        existing_labels.resize(initial_points);
    }

    // 重写标签文件：包含旧标签和新合并的标签
    {
        std::ofstream label_writer(label_file_path, std::ios::trunc);
        if (!label_writer.is_open()) {
             throw ANNException("无法打开标签文件进行写入", -1, __FUNCSIG__, __FILE__, __LINE__);
        }
        for (const auto& lbl : existing_labels) {
            label_writer << lbl << "\n";
        }
        for (size_t i = 0; i < num_active_points; ++i) {
             label_writer << mem_tags[i] << "\n";
        }
        label_writer.close();
    }

    // 释放旧磁盘索引句柄以解开文件锁
    _disk_index.reset(); 

    // 删除旧的 PQ 文件以强制使用新的块数重新生成
    std::string pq_pivots_path = _disk_index_path + "_pq_pivots.bin";
    std::string pq_compressed_path = _disk_index_path + "_pq_compressed.bin";
    std::remove(pq_pivots_path.c_str());
    std::remove(pq_compressed_path.c_str());
    
    // 调用 DiskANN 核心 API 重新构建磁盘索引
    int res = diskann::build_disk_index<T, LabelT>(
        _data_file_path.c_str(),
        _disk_index_path.c_str(),
        params.c_str(),
        diskann::Metric::L2,
        false, // use_opq
        "",    // codebook_prefix
        true,  // use_filters (使用标签文件)
        label_file_path, 
        "" 
    );
    
    if (res != 0) {
        throw ANNException("合并期间重建磁盘索引失败", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    // 修复：build_disk_index 会调用 convert_labels_string_to_int 从而破坏数值标签
    // 我们需要用正确的标签文件覆盖 build_disk_index 生成的错误标签文件
    // 假设 build_disk_index 保持了点的原始顺序（merge_shards 似乎是这样做的）
    std::string bad_label_file = _disk_index_path + "_disk.index_labels.txt";
    copy_file(label_file_path, bad_label_file);

    // 重新加载索引并重置内存状态
    load_disk_index();
    _mem_index->init_empty_index();
    
    // 更新删除状态：保留全局删除记录，并更新磁盘删除 ID 集合
    _disk_deleted_ids.clear();
    for (const auto& label : _deleted_labels) {
        if (_disk_label_to_id.find(label) != _disk_label_to_id.end()) {
            _disk_deleted_ids.insert(_disk_label_to_id[label]);
        }
    }

    // 清理临时文件
    std::remove(temp_mem_index_path.c_str());
    std::remove(temp_mem_data_path.c_str());
    std::remove(temp_mem_tags_path.c_str());

    diskann::cout << "合并成功完成。" << std::endl;
}

// 模板类显式实例化
template class DynamicDiskIndex<float, uint32_t>;
template class DynamicDiskIndex<int8_t, uint32_t>;
template class DynamicDiskIndex<uint8_t, uint32_t>;

} // namespace diskann
