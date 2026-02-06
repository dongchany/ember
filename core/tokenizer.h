#pragma once

#include "error.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <limits>
#include <cctype>
#include <cstring>
#include <regex>

namespace ember {

// Tokenizer 接口
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    
    virtual std::vector<int> encode(const std::string& text, 
                                    bool add_special_tokens = true) const = 0;
    virtual std::string decode(const std::vector<int>& tokens,
                               bool skip_special_tokens = true) const = 0;
    
    virtual int bos_token_id() const = 0;
    virtual int eos_token_id() const = 0;
    virtual int pad_token_id() const = 0;
    virtual bool is_special_token(int token_id) const = 0;
    virtual size_t vocab_size() const = 0;
};

// HuggingFace Tokenizer（从 tokenizer.json 加载）
class HFTokenizer : public ITokenizer {
public:
    HFTokenizer() = default;
    
    Error load(const std::string& model_dir) {
        std::string tokenizer_path = model_dir + "/tokenizer.json";
        std::ifstream f(tokenizer_path);
        if (!f.is_open()) {
            return Error::file_not_found(tokenizer_path);
        }
        
        std::stringstream buffer;
        buffer << f.rdbuf();
        std::string content = buffer.str();
        
        Error err = parse_tokenizer_json(content);
        if (err) return err;

        if (bpe_ranks_.empty()) {
            load_merges_txt(model_dir);
        }
        
        load_special_tokens(model_dir);
        
        return Error::success();
    }
    
    std::vector<int> encode(const std::string& text, 
                            bool add_special_tokens = true) const override {
        std::vector<int> tokens;
        
        if (add_special_tokens && add_bos_token_ && bos_token_id_ >= 0) {
            tokens.push_back(bos_token_id_);
        }
        
        if (token_to_id_.empty()) {
            return tokens;
        }
        
        auto encode_segment = [&](const std::string& segment) {
            if (segment.empty()) return;
            std::vector<std::string> pieces = split_by_regex(segment);
            for (const auto& piece : pieces) {
                if (piece.empty()) continue;
                std::string encoded = byte_encode(piece);
                std::vector<std::string> bpe_tokens = bpe(encoded);
                for (const auto& bpe_token : bpe_tokens) {
                    auto it = token_to_id_.find(bpe_token);
                    if (it != token_to_id_.end()) {
                        tokens.push_back(it->second);
                    } else if (unk_token_id_ >= 0) {
                        tokens.push_back(unk_token_id_);
                    }
                }
            }
        };
        
        if (special_token_strings_.empty()) {
            encode_segment(text);
        } else {
            size_t i = 0;
            while (i < text.size()) {
                size_t match_len = 0;
                int match_id = -1;
                for (const auto& s : special_token_strings_) {
                    if (s.size() > match_len && text.compare(i, s.size(), s) == 0) {
                        auto it = token_to_id_.find(s);
                        if (it != token_to_id_.end()) {
                            match_len = s.size();
                            match_id = it->second;
                        }
                    }
                }
                
                if (match_id >= 0) {
                    tokens.push_back(match_id);
                    i += match_len;
                    continue;
                }
                
                size_t next = text.size();
                for (const auto& s : special_token_strings_) {
                    size_t pos = text.find(s, i);
                    if (pos != std::string::npos && pos < next) {
                        next = pos;
                    }
                }
                
                encode_segment(text.substr(i, next - i));
                i = next;
            }
        }
        
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens,
                       bool skip_special_tokens = true) const override {
        std::string text;
        
        for (int id : tokens) {
            if (skip_special_tokens && special_tokens_.count(id)) {
                continue;
            }
            
            auto it = id_to_token_.find(id);
            if (it != id_to_token_.end()) {
                text += it->second;
            }
        }
        
        return byte_decode(text);
    }
    
    int bos_token_id() const override { return bos_token_id_; }
    int eos_token_id() const override { return eos_token_id_; }
    int pad_token_id() const override { return pad_token_id_; }
    bool is_special_token(int token_id) const override { 
        return special_tokens_.count(token_id) > 0; 
    }
    size_t vocab_size() const override { return id_to_token_.size(); }

private:
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;
    std::set<int> special_tokens_;
    std::unordered_map<std::string, int> bpe_ranks_;
    mutable std::unordered_map<std::string, std::vector<std::string>> bpe_cache_;
    std::array<std::string, 256> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;
    std::vector<std::string> special_token_strings_;
    
    int bos_token_id_ = 151643;
    int eos_token_id_ = 151645;
    int pad_token_id_ = 151643;
    int unk_token_id_ = -1;
    bool byte_encoder_ready_ = false;
    bool add_bos_token_ = true;
    
    static std::string utf8_encode(uint32_t codepoint) {
        std::string out;
        if (codepoint <= 0x7F) {
            out.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else if (codepoint <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
        }
        return out;
    }
    
    static bool is_all_space(const std::string& s) {
        for (unsigned char c : s) {
            if (!std::isspace(c)) return false;
        }
        return !s.empty();
    }
    
    void init_byte_encoder() {
        if (byte_encoder_ready_) return;
        std::vector<int> bs;
        bs.reserve(256);
        for (int i = 33; i <= 126; ++i) bs.push_back(i);
        for (int i = 161; i <= 172; ++i) bs.push_back(i);
        for (int i = 174; i <= 255; ++i) bs.push_back(i);
        
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(256 + n);
                n++;
            }
        }
        
        for (size_t i = 0; i < bs.size(); ++i) {
            std::string s = utf8_encode(static_cast<uint32_t>(cs[i]));
            byte_encoder_[static_cast<unsigned char>(bs[i])] = s;
            byte_decoder_[s] = static_cast<unsigned char>(bs[i]);
        }
        
        byte_encoder_ready_ = true;
    }
    
    std::string byte_encode(const std::string& text) const {
        std::string out;
        for (unsigned char c : text) {
            out += byte_encoder_[c];
        }
        return out;
    }
    
    std::string byte_decode(const std::string& text) const {
        std::string out;
        auto chars = split_utf8(text);
        for (const auto& ch : chars) {
            auto it = byte_decoder_.find(ch);
            if (it != byte_decoder_.end()) {
                out.push_back(static_cast<char>(it->second));
            } else {
                out += ch;
            }
        }
        return out;
    }
    
    std::vector<std::string> split_on_spaces(const std::string& text) const {
        std::vector<std::string> parts;
        size_t i = 0;
        while (i < text.size()) {
            bool space = std::isspace(static_cast<unsigned char>(text[i]));
            size_t j = i + 1;
            while (j < text.size() && std::isspace(static_cast<unsigned char>(text[j])) == space) {
                j++;
            }
            parts.push_back(text.substr(i, j - i));
            i = j;
        }
        
        std::vector<std::string> merged;
        for (size_t idx = 0; idx < parts.size(); ++idx) {
            if (parts[idx] == " " &&
                idx + 1 < parts.size() && !is_all_space(parts[idx + 1])) {
                merged.push_back(parts[idx] + parts[idx + 1]);
                idx++;
            } else {
                merged.push_back(parts[idx]);
            }
        }
        
        return merged;
    }

    std::vector<std::string> split_by_regex(const std::string& text) const {
        static const std::regex pattern(
            R"('s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+)"
        );
        
        std::vector<std::string> tokens;
        try {
            auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
            auto end = std::sregex_iterator();
            for (auto it = begin; it != end; ++it) {
                tokens.push_back(it->str());
            }
        } catch (const std::regex_error&) {
            return split_on_spaces(text);
        }
        
        if (tokens.empty()) {
            return split_on_spaces(text);
        }
        return tokens;
    }
    
    static std::vector<std::string> split_utf8(const std::string& s) {
        std::vector<std::string> out;
        for (size_t i = 0; i < s.size(); ) {
            unsigned char c = static_cast<unsigned char>(s[i]);
            size_t len = 1;
            if ((c & 0x80) == 0x00) {
                len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                len = 4;
            }
            out.push_back(s.substr(i, len));
            i += len;
        }
        return out;
    }
    
    static std::string pair_key(const std::string& a, const std::string& b) {
        return a + "\n" + b;
    }
    
    void normalize_special_token_strings() {
        std::sort(special_token_strings_.begin(), special_token_strings_.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
        std::unordered_set<std::string> seen;
        std::vector<std::string> dedup;
        dedup.reserve(special_token_strings_.size());
        for (const auto& s : special_token_strings_) {
            if (seen.insert(s).second) {
                dedup.push_back(s);
            }
        }
        special_token_strings_.swap(dedup);
    }
    
    std::vector<std::string> bpe(const std::string& token) const {
        auto it_cache = bpe_cache_.find(token);
        if (it_cache != bpe_cache_.end()) {
            return it_cache->second;
        }
        
        std::vector<std::string> word = split_utf8(token);
        if (word.size() <= 1) {
            bpe_cache_[token] = word;
            return word;
        }
        
        auto get_pairs = [](const std::vector<std::string>& w) {
            std::vector<std::pair<std::string, std::string>> pairs;
            pairs.reserve(w.size());
            for (size_t i = 0; i + 1 < w.size(); ++i) {
                pairs.emplace_back(w[i], w[i + 1]);
            }
            return pairs;
        };
        
        std::vector<std::pair<std::string, std::string>> pairs = get_pairs(word);
        while (true) {
            int best_rank = std::numeric_limits<int>::max();
            std::pair<std::string, std::string> best_pair;
            bool found = false;
            
            for (const auto& pair : pairs) {
                auto it = bpe_ranks_.find(pair_key(pair.first, pair.second));
                if (it != bpe_ranks_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pair = pair;
                    found = true;
                }
            }
            
            if (!found) break;
            
            std::vector<std::string> new_word;
            new_word.reserve(word.size());
            for (size_t i = 0; i < word.size(); ) {
                if (i + 1 < word.size() && word[i] == best_pair.first && word[i + 1] == best_pair.second) {
                    new_word.push_back(word[i] + word[i + 1]);
                    i += 2;
                } else {
                    new_word.push_back(word[i]);
                    i += 1;
                }
            }
            word.swap(new_word);
            if (word.size() <= 1) break;
            pairs = get_pairs(word);
        }
        
        bpe_cache_[token] = word;
        return word;
    }
    
    static void skip_ws(const char*& p, const char* end) {
        while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    }
    
    static bool parse_str(const char*& p, const char* end, std::string& out) {
        skip_ws(p, end);
        if (p >= end || *p != '"') return false;
        p++;
        out.clear();
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                p++;
                switch (*p) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    case 'u': {
                        if (p + 4 < end) {
                            char hex[5] = {p[1], p[2], p[3], p[4], 0};
                            int code = strtol(hex, nullptr, 16);
                            if (code < 0x80) {
                                out += static_cast<char>(code);
                            } else if (code < 0x800) {
                                out += static_cast<char>(0xC0 | (code >> 6));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            } else {
                                out += static_cast<char>(0xE0 | (code >> 12));
                                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            p += 4;
                        }
                        break;
                    }
                    default: out += *p; break;
                }
            } else {
                out += *p;
            }
            p++;
        }
        if (p < end) p++;
        return true;
    }
    
    static bool parse_int(const char*& p, const char* end, int& out) {
        skip_ws(p, end);
        if (p >= end) return false;
        bool neg = (*p == '-');
        if (neg) p++;
        if (p >= end || !isdigit(*p)) return false;
        out = 0;
        while (p < end && isdigit(*p)) {
            out = out * 10 + (*p - '0');
            p++;
        }
        if (neg) out = -out;
        return true;
    }
    
    Error parse_tokenizer_json(const std::string& content) {
        id_to_token_.clear();
        token_to_id_.clear();
        bpe_ranks_.clear();
        bpe_cache_.clear();
        special_token_strings_.clear();
        special_tokens_.clear();
        byte_decoder_.clear();
        byte_encoder_ready_ = false;
        
        Error err = parse_vocab_from_tokenizer_json(content);
        if (err) return err;
        parse_merges_from_tokenizer_json(content);
        parse_added_tokens(content);
        init_byte_encoder();
        return Error::success();
    }
    
    Error parse_vocab_from_tokenizer_json(const std::string& content) {
        // 查找 "model" 下的 "vocab"
        size_t model_pos = content.find("\"model\"");
        if (model_pos == std::string::npos) {
            return Error(ErrorCode::INVALID_FORMAT, "model not found");
        }
        size_t vocab_pos = content.find("\"vocab\"", model_pos);
        if (vocab_pos == std::string::npos) {
            return Error(ErrorCode::INVALID_FORMAT, "vocab not found");
        }
        
        size_t start = content.find('{', vocab_pos + 7);
        if (start == std::string::npos) {
            return Error(ErrorCode::INVALID_FORMAT, "vocab object not found");
        }
        
        const char* p = content.c_str() + start + 1;
        const char* end = content.c_str() + content.size();
        
        while (p < end) {
            skip_ws(p, end);
            if (*p == '}') break;
            
            std::string token;
            if (!parse_str(p, end, token)) break;
            
            skip_ws(p, end);
            if (p >= end || *p != ':') break;
            p++;
            
            int id;
            if (!parse_int(p, end, id)) break;
            
            id_to_token_[id] = token;
            token_to_id_[token] = id;
            
            skip_ws(p, end);
            if (*p == ',') p++;
        }
        
        return Error::success();
    }
    
    void parse_merges_from_tokenizer_json(const std::string& content) {
        size_t model_pos = content.find("\"model\"");
        if (model_pos == std::string::npos) return;
        size_t merges_pos = content.find("\"merges\"", model_pos);
        if (merges_pos == std::string::npos) return;
        
        size_t start = content.find('[', merges_pos + 8);
        if (start == std::string::npos) return;
        
        const char* p = content.c_str() + start + 1;
        const char* end = content.c_str() + content.size();
        
        int rank = 0;
        while (p < end) {
            skip_ws(p, end);
            if (p >= end || *p == ']') break;
            
            if (*p == '[') {
                p++;
                std::string a;
                std::string b;
                if (!parse_str(p, end, a)) break;
                skip_ws(p, end);
                if (p < end && *p == ',') p++;
                if (!parse_str(p, end, b)) break;
                while (p < end && *p != ']') p++;
                if (p < end && *p == ']') p++;
                if (!a.empty() && !b.empty()) {
                    bpe_ranks_[pair_key(a, b)] = rank++;
                }
            } else if (*p == '"') {
                std::string merge;
                if (!parse_str(p, end, merge)) break;
                size_t sep = merge.find(' ');
                if (sep != std::string::npos) {
                    std::string a = merge.substr(0, sep);
                    std::string b = merge.substr(sep + 1);
                    if (!a.empty() && !b.empty()) {
                        bpe_ranks_[pair_key(a, b)] = rank++;
                    }
                }
            } else {
                p++;
            }
            
            skip_ws(p, end);
            if (p < end && *p == ',') p++;
        }
    }
    
    void parse_added_tokens(const std::string& content) {
        size_t added_pos = content.find("\"added_tokens\"");
        if (added_pos == std::string::npos) return;
        size_t start = content.find('[', added_pos + 13);
        if (start == std::string::npos) return;
        
        const char* p = content.c_str() + start + 1;
        const char* end = content.c_str() + content.size();
        
        while (p < end) {
            skip_ws(p, end);
            if (*p == ']') break;
            if (*p != '{') {
                p++;
                continue;
            }
            p++;
            
            int id = -1;
            std::string content_str;
            bool is_special = false;
            
            while (p < end && *p != '}') {
                skip_ws(p, end);
                
                std::string key;
                if (!parse_str(p, end, key)) break;
                if (p >= end || *p != ':') break;
                p++;
                
                if (key == "id") {
                    int value = -1;
                    if (parse_int(p, end, value)) id = value;
                } else if (key == "content") {
                    std::string value;
                    if (parse_str(p, end, value)) content_str = value;
                } else if (key == "special") {
                    skip_ws(p, end);
                    if (p + 3 < end && std::strncmp(p, "true", 4) == 0) {
                        is_special = true;
                        p += 4;
                    } else if (p + 4 < end && std::strncmp(p, "false", 5) == 0) {
                        is_special = false;
                        p += 5;
                    }
                } else {
                    skip_ws(p, end);
                    if (*p == '"') {
                        std::string dummy;
                        parse_str(p, end, dummy);
                    } else if (*p == '{') {
                        int depth = 1;
                        p++;
                        while (p < end && depth > 0) {
                            if (*p == '{') depth++;
                            else if (*p == '}') depth--;
                            p++;
                        }
                    } else {
                        while (p < end && *p != ',' && *p != '}') p++;
                    }
                }
                
                skip_ws(p, end);
                if (*p == ',') p++;
            }
            
            if (*p == '}') p++;
            if (id >= 0 && !content_str.empty()) {
                id_to_token_[id] = content_str;
                token_to_id_[content_str] = id;
                if (is_special) {
                    special_tokens_.insert(id);
                    special_token_strings_.push_back(content_str);
                }
            }
            
            skip_ws(p, end);
            if (*p == ',') p++;
        }
        
        normalize_special_token_strings();
    }

    void load_merges_txt(const std::string& model_dir) {
        std::string merges_path = model_dir + "/merges.txt";
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            return;
        }
        
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sep = line.find(' ');
            if (sep == std::string::npos) continue;
            std::string a = line.substr(0, sep);
            std::string b = line.substr(sep + 1);
            if (a.empty() || b.empty()) continue;
            bpe_ranks_[pair_key(a, b)] = rank++;
        }
    }
    
    void load_special_tokens(const std::string& model_dir) {
        std::string gen_path = model_dir + "/generation_config.json";
        std::string tok_path = model_dir + "/tokenizer_config.json";
        
        std::string gen_content;
        std::string tok_content;
        
        {
            std::ifstream f(gen_path);
            if (f.is_open()) {
                std::stringstream buffer;
                buffer << f.rdbuf();
                gen_content = buffer.str();
            }
        }
        {
            std::ifstream f(tok_path);
            if (f.is_open()) {
                std::stringstream buffer;
                buffer << f.rdbuf();
                tok_content = buffer.str();
            }
        }
        
        auto find_id = [](const std::string& content, const std::string& key) -> int {
            if (content.empty()) return -1;
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return -1;
            pos = content.find(':', pos);
            if (pos == std::string::npos) return -1;
            pos++;
            while (pos < content.size() && !isdigit(content[pos]) && content[pos] != '-') pos++;
            if (pos >= content.size()) return -1;
            return std::stoi(content.substr(pos));
        };
        
        auto find_bool = [](const std::string& content, const std::string& key, bool default_value) -> bool {
            if (content.empty()) return default_value;
            size_t pos = content.find("\"" + key + "\"");
            if (pos == std::string::npos) return default_value;
            pos = content.find(':', pos);
            if (pos == std::string::npos) return default_value;
            pos++;
            while (pos < content.size() && (content[pos] == ' ' || content[pos] == '\t')) pos++;
            if (content.compare(pos, 4, "true") == 0) return true;
            if (content.compare(pos, 5, "false") == 0) return false;
            return default_value;
        };
        
        int bos = find_id(gen_content, "bos_token_id");
        int eos = find_id(gen_content, "eos_token_id");
        int pad = find_id(gen_content, "pad_token_id");
        if (bos < 0) bos = find_id(tok_content, "bos_token_id");
        if (eos < 0) eos = find_id(tok_content, "eos_token_id");
        if (pad < 0) pad = find_id(tok_content, "pad_token_id");
        
        // add_bos_token 仅从 tokenizer_config.json 获取
        add_bos_token_ = find_bool(tok_content, "add_bos_token", add_bos_token_);
        
        if (bos > 0) bos_token_id_ = bos;
        if (eos > 0) eos_token_id_ = eos;
        if (pad > 0) pad_token_id_ = pad;
        
        // 添加特殊 tokens
        special_tokens_.insert(bos_token_id_);
        special_tokens_.insert(eos_token_id_);
        special_tokens_.insert(pad_token_id_);
        
        // Qwen3 特殊 tokens 范围
        for (int id = 151643; id <= 151665; ++id) {
            special_tokens_.insert(id);
        }
        
        if (id_to_token_.count(bos_token_id_)) {
            special_token_strings_.push_back(id_to_token_[bos_token_id_]);
        }
        if (id_to_token_.count(eos_token_id_)) {
            special_token_strings_.push_back(id_to_token_[eos_token_id_]);
        }
        if (id_to_token_.count(pad_token_id_)) {
            special_token_strings_.push_back(id_to_token_[pad_token_id_]);
        }
        normalize_special_token_strings();
    }
};

// 简单回退 Tokenizer
class SimpleTokenizer : public ITokenizer {
public:
    SimpleTokenizer() {
        bos_token_id_ = 151643;
        eos_token_id_ = 151645;
        pad_token_id_ = 151643;
    }
    
    std::vector<int> encode(const std::string& text, 
                            bool add_special_tokens = true) const override {
        std::vector<int> tokens;
        if (add_special_tokens) {
            tokens.push_back(bos_token_id_);
        }
        return tokens;
    }
    
    std::string decode(const std::vector<int>& tokens,
                       bool skip_special_tokens = true) const override {
        std::string result;
        for (int id : tokens) {
            if (!skip_special_tokens || !is_special_token(id)) {
                result += "[" + std::to_string(id) + "]";
            }
        }
        return result;
    }
    
    int bos_token_id() const override { return bos_token_id_; }
    int eos_token_id() const override { return eos_token_id_; }
    int pad_token_id() const override { return pad_token_id_; }
    bool is_special_token(int token_id) const override { 
        return token_id >= 151643 && token_id <= 151665;
    }
    size_t vocab_size() const override { return 151936; }

private:
    int bos_token_id_;
    int eos_token_id_;
    int pad_token_id_;
};

}  // namespace ember
