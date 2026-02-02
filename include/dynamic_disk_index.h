#pragma once

#include "index.h"
#include "pq_flash_index.h"
#include <vector>
#include <shared_mutex>
#include <atomic>
#include "tsl/robin_set.h"
#include "tsl/robin_map.h"

namespace diskann {

template<typename T, typename LabelT = uint32_t>
class DynamicDiskIndex {
public:
    DISKANN_DLLEXPORT DynamicDiskIndex(const IndexConfig& config, const std::string& data_file_path, const std::string& disk_index_path, size_t mem_index_threshold);
    DISKANN_DLLEXPORT ~DynamicDiskIndex();

    // 插入
    DISKANN_DLLEXPORT void insert(const T* point, const LabelT label);
    
    // 删除
    DISKANN_DLLEXPORT void remove(const LabelT label);

    // 搜索
    DISKANN_DLLEXPORT void search(const T* query, size_t k, size_t l, uint64_t* indices, float* distances);

    // 合并
    DISKANN_DLLEXPORT void merge();

private:
    std::shared_ptr<Index<T, LabelT>> _mem_index;
    std::shared_ptr<PQFlashIndex<T, LabelT>> _disk_index;
    
    // 删除集合 (存储 LabelT)
    tsl::robin_set<LabelT> _deleted_labels;
    
    // 磁盘索引的删除集合 (存储内部 ID)
    tsl::robin_map<LabelT, uint32_t> _disk_label_to_id;
    tsl::robin_set<uint32_t> _disk_deleted_ids;

    std::shared_mutex _rw_lock;
    std::string _data_file_path;
    std::string _disk_index_path;
    IndexConfig _config;
    size_t _mem_index_threshold;
    
    // 辅助函数：加载磁盘索引并构建 Label -> ID 映射
    void load_disk_index();
};

}
