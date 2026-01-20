#include "safetensors.h"
#include <fstream>
#include <cstring>
#include <filesystem>
#include <algorithm>
#include <iostream>

// 简单的 JSON 解析（只支持 safetensors header 格式）
// 注意：这是一个简化实现，生产环境建议使用 nlohmann/json 等库

namespace ember {

namespace {

// 跳过空白字符
void skip_whitespace(const char*& p, const char* end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) {
        ++p;
    }
}

// 解析字符串（带引号）
bool parse_string(const char*& p, const char* end, std::string& out) {
    skip_whitespace(p, end);
    if (p >= end || *p != '"') return false;
    ++p;  // 跳过开始引号
    
    out.clear();
    while (p < end && *p != '"') {
        if (*p == '\\' && p + 1 < end) {
            ++p;
            switch (*p) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case '\\': out += '\\'; break;
                case '"': out += '"'; break;
                default: out += *p; break;
            }
        } else {
            out += *p;
        }
        ++p;
    }
    
    if (p >= end) return false;
    ++p;  // 跳过结束引号
    return true;
}

// 解析数字
bool parse_number(const char*& p, const char* end, int64_t& out) {
    skip_whitespace(p, end);
    if (p >= end) return false;
    
    bool negative = false;
    if (*p == '-') {
        negative = true;
        ++p;
    }
    
    if (p >= end || !std::isdigit(*p)) return false;
    
    out = 0;
    while (p < end && std::isdigit(*p)) {
        out = out * 10 + (*p - '0');
        ++p;
    }
    
    if (negative) out = -out;
    return true;
}

// 跳过到指定字符
bool skip_to(const char*& p, const char* end, char c) {
    skip_whitespace(p, end);
    if (p >= end || *p != c) return false;
    ++p;
    return true;
}

}  // namespace

SafetensorsReader::~SafetensorsReader() {
    close();
}

void SafetensorsReader::close() {
    if (file_.is_open()) {
        file_.close();
    }
    tensors_.clear();
    
    // TODO: 如果使用了 mmap，需要 munmap
}

Error SafetensorsReader::open(const std::string& path) {
    path_ = path;
    
    // 打开文件
    file_.open(path, std::ios::binary);
    if (!file_.is_open()) {
        return Error::file_not_found(path);
    }
    
    // 获取文件大小
    file_.seekg(0, std::ios::end);
    file_size_ = file_.tellg();
    file_.seekg(0, std::ios::beg);
    
    if (file_size_ < 8) {
        return Error(ErrorCode::INVALID_FORMAT, "File too small");
    }
    
    // 解析头部
    return parse_header();
}

Error SafetensorsReader::parse_header() {
    // 读取头部大小（8 字节小端序 uint64）
    uint64_t header_len;
    file_.read(reinterpret_cast<char*>(&header_len), 8);
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read header size");
    }
    
    header_size_ = header_len;
    data_offset_ = 8 + header_len;
    data_size_ = file_size_ - data_offset_;
    
    if (header_len > 100 * 1024 * 1024) {  // 100MB 限制
        return Error(ErrorCode::INVALID_FORMAT, "Header too large");
    }
    
    // 读取 JSON 头部
    std::string header(header_len, '\0');
    file_.read(&header[0], header_len);
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read header");
    }
    
    // 解析 JSON
    const char* p = header.c_str();
    const char* end = p + header.size();
    
    skip_whitespace(p, end);
    if (p >= end || *p != '{') {
        return Error(ErrorCode::INVALID_FORMAT, "Expected '{' at start of header");
    }
    p++;  // 消耗 {
    
    // 解析每个张量条目
    while (p < end) {
        skip_whitespace(p, end);
        if (*p == '}') break;
        
        // 解析张量名
        std::string name;
        if (!parse_string(p, end, name)) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected tensor name");
        }
        
        // 跳过 ':'
        if (!skip_to(p, end, ':')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected ':'");
        }
        
        // 跳过 '__metadata__' 条目
        if (name == "__metadata__") {
            // 跳过整个对象
            skip_whitespace(p, end);
            if (p < end && *p == '{') {
                int depth = 1;
                p++;  // 消耗开始的 {
                bool in_string = false;
                while (p < end && depth > 0) {
                    if (!in_string) {
                        if (*p == '{') depth++;
                        else if (*p == '}') depth--;
                        else if (*p == '"') in_string = true;
                    } else {
                        if (*p == '\\' && p + 1 < end) p++;
                        else if (*p == '"') in_string = false;
                    }
                    p++;
                }
            }
            skip_whitespace(p, end);
            if (p < end && *p == ',') p++;
            continue;
        }
        
        SafetensorsMeta meta;
        meta.name = name;
        
        // 解析张量元数据对象
        if (!skip_to(p, end, '{')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected tensor metadata object");
        }
        
        while (p < end && *p != '}') {
            skip_whitespace(p, end);
            
            std::string key;
            if (!parse_string(p, end, key)) break;
            
            if (!skip_to(p, end, ':')) {
                return Error(ErrorCode::INVALID_FORMAT, "Expected ':'");
            }
            
            if (key == "dtype") {
                std::string dtype_str;
                if (!parse_string(p, end, dtype_str)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected dtype string");
                }
                meta.dtype = safetensors_dtype_to_ember(dtype_str);
            } else if (key == "shape") {
                if (!skip_to(p, end, '[')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected shape array");
                }
                
                while (p < end && *p != ']') {
                    skip_whitespace(p, end);
                    if (*p == ']') break;
                    
                    int64_t dim;
                    if (!parse_number(p, end, dim)) break;
                    meta.shape.push_back(dim);
                    
                    skip_whitespace(p, end);
                    if (*p == ',') p++;
                }
                
                if (!skip_to(p, end, ']')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ']'");
                }
            } else if (key == "data_offsets") {
                if (!skip_to(p, end, '[')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected data_offsets array");
                }
                
                int64_t start_offset, end_offset;
                if (!parse_number(p, end, start_offset)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected start offset");
                }
                
                skip_whitespace(p, end);
                if (!skip_to(p, end, ',')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ','");
                }
                
                if (!parse_number(p, end, end_offset)) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected end offset");
                }
                
                meta.data_offset = start_offset;
                meta.data_size = end_offset - start_offset;
                
                if (!skip_to(p, end, ']')) {
                    return Error(ErrorCode::INVALID_FORMAT, "Expected ']'");
                }
            }
            
            skip_whitespace(p, end);
            if (*p == ',') p++;
        }
        
        if (!skip_to(p, end, '}')) {
            return Error(ErrorCode::INVALID_FORMAT, "Expected '}'");
        }
        
        tensors_[name] = std::move(meta);
        
        skip_whitespace(p, end);
        if (*p == ',') p++;
    }
    
    return Error::success();
}

Error SafetensorsReader::read_tensor(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    
    const auto& meta = it->second;
    if (dst_size < meta.data_size) {
        return Error(ErrorCode::INVALID_ARGUMENT, "Buffer too small");
    }
    
    file_.seekg(data_offset_ + meta.data_offset);
    file_.read(static_cast<char*>(dst), meta.data_size);
    
    if (!file_) {
        return Error(ErrorCode::FILE_READ_ERROR, "Failed to read tensor data");
    }
    
    return Error::success();
}

Error SafetensorsReader::read_tensor(const std::string& name, Tensor& out) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    
    const auto& meta = it->second;
    
    // 分配内存
    void* data = malloc(meta.data_size);
    if (!data) {
        return Error::out_of_memory("Failed to allocate tensor memory");
    }
    
    // 读取数据
    Error err = read_tensor(name, data, meta.data_size);
    if (err) {
        free(data);
        return err;
    }
    
    out.shape = meta.shape;
    out.dtype = meta.dtype;
    out.data = data;
    out.device_id = DEVICE_CPU;
    
    return Error::success();
}

const void* SafetensorsReader::get_data_ptr(const std::string& name) const {
    // 如果使用 mmap，返回映射地址
    if (mmap_ptr_) {
        auto it = tensors_.find(name);
        if (it != tensors_.end()) {
            return static_cast<const char*>(mmap_ptr_) + data_offset_ + it->second.data_offset;
        }
    }
    return nullptr;
}

std::vector<std::string> SafetensorsReader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, _] : tensors_) {
        names.push_back(name);
    }
    return names;
}

// ModelWeightLoader 实现

Error ModelWeightLoader::open(const std::string& model_dir) {
    model_dir_ = model_dir;
    namespace fs = std::filesystem;
    
    if (!fs::exists(model_dir)) {
        return Error::file_not_found(model_dir);
    }
    
    // 查找所有 .safetensors 文件
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(model_dir)) {
        if (entry.path().extension() == ".safetensors") {
            files.push_back(entry.path().string());
        }
    }
    
    if (files.empty()) {
        return Error(ErrorCode::FILE_NOT_FOUND, "No safetensors files found in " + model_dir);
    }
    
    // 排序以保证顺序一致
    std::sort(files.begin(), files.end());
    
    // 打开每个文件
    for (size_t i = 0; i < files.size(); ++i) {
        auto reader = std::make_unique<SafetensorsReader>();
        Error err = reader->open(files[i]);
        if (err) {
            return err;
        }
        
        // 记录每个张量在哪个文件
        for (const auto& name : reader->tensor_names()) {
            tensor_to_file_[name] = static_cast<int>(i);
        }
        
        readers_.push_back(std::move(reader));
    }
    
    return Error::success();
}

void ModelWeightLoader::close() {
    readers_.clear();
    tensor_to_file_.clear();
}

std::vector<std::string> ModelWeightLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensor_to_file_.size());
    for (const auto& [name, _] : tensor_to_file_) {
        names.push_back(name);
    }
    return names;
}

bool ModelWeightLoader::has_tensor(const std::string& name) const {
    return tensor_to_file_.count(name) > 0;
}

const SafetensorsMeta* ModelWeightLoader::get_meta(const std::string& name) const {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) return nullptr;
    return readers_[it->second]->get_meta(name);
}

Error ModelWeightLoader::read_tensor(const std::string& name, void* dst, size_t dst_size) {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    return readers_[it->second]->read_tensor(name, dst, dst_size);
}

Error ModelWeightLoader::read_tensor(const std::string& name, Tensor& out) {
    auto it = tensor_to_file_.find(name);
    if (it == tensor_to_file_.end()) {
        return Error(ErrorCode::WEIGHT_NOT_FOUND, "Tensor not found: " + name);
    }
    return readers_[it->second]->read_tensor(name, out);
}

}  // namespace ember
