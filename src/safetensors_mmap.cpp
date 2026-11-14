// safetensors_mmap.cpp -- see safetensors_mmap.h. The header is the safetensors format: an 8-byte little-endian length, the
// JSON object of tensors (dtype, shape, data_offsets [begin, end) from the end of the header), then the data.
#include "safetensors_mmap.h"

#include <dirent.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {

std::string g_error;

// ---- a small JSON reader: enough for the headers and the index (objects, arrays, strings, numbers, true/false/null)
struct Json {
    enum Kind { Null, Bool, Num, Str, Arr, Obj } kind = Null;
    double num = 0;
    std::string str;
    std::vector<Json> arr;
    std::vector<std::pair<std::string, Json>> obj;  // in file order
    const Json *get(const char *k) const {
        for (const auto &kv : obj)
            if (kv.first == k) return &kv.second;
        return nullptr;
    }
};

struct JsonParser {
    const char *p, *end;
    bool ok = true;
    void ws() {
        while (p < end && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) ++p;
    }
    bool lit(const char *s) {
        const size_t n = strlen(s);
        if ((size_t)(end - p) < n || memcmp(p, s, n) != 0) return false;
        p += n;
        return true;
    }
    std::string string() {
        std::string out;
        if (p >= end || *p != '"') { ok = false; return out; }
        ++p;
        while (p < end && *p != '"') {
            if (*p == '\\' && p + 1 < end) {
                ++p;
                switch (*p) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {  // \uXXXX: kept as UTF-8 for the BMP (the names here are ASCII)
                    if (end - p < 5) { ok = false; return out; }
                    unsigned cp = (unsigned)strtoul(std::string(p + 1, 4).c_str(), nullptr, 16);
                    if (cp < 0x80) out += (char)cp;
                    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
                    else { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
                    p += 4;
                    break;
                }
                default: out += *p;
                }
                ++p;
            } else {
                out += *p++;
            }
        }
        if (p >= end) { ok = false; return out; }
        ++p;
        return out;
    }
    Json value(int depth = 0) {
        Json j;
        ws();
        if (p >= end || depth > 64) { ok = false; return j; }
        switch (*p) {
        case '{': {
            j.kind = Json::Obj;
            ++p;
            ws();
            if (p < end && *p == '}') { ++p; return j; }
            for (;;) {
                ws();
                std::string k = string();
                ws();
                if (!ok || p >= end || *p != ':') { ok = false; return j; }
                ++p;
                Json v = value(depth + 1);
                if (!ok) return j;
                j.obj.emplace_back(std::move(k), std::move(v));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == '}') { ++p; return j; }
                ok = false;
                return j;
            }
        }
        case '[': {
            j.kind = Json::Arr;
            ++p;
            ws();
            if (p < end && *p == ']') { ++p; return j; }
            for (;;) {
                Json v = value(depth + 1);
                if (!ok) return j;
                j.arr.push_back(std::move(v));
                ws();
                if (p < end && *p == ',') { ++p; continue; }
                if (p < end && *p == ']') { ++p; return j; }
                ok = false;
                return j;
            }
        }
        case '"': j.kind = Json::Str; j.str = string(); return j;
        case 't': if (lit("true")) { j.kind = Json::Bool; j.num = 1; } else ok = false; return j;
        case 'f': if (lit("false")) { j.kind = Json::Bool; } else ok = false; return j;
        case 'n': if (lit("null")) { j.kind = Json::Null; } else ok = false; return j;
        default: {
            char *e = nullptr;
            j.num = strtod(p, &e);
            if (e == p || e > end) { ok = false; return j; }
            p = e;
            j.kind = Json::Num;
            return j;
        }
        }
    }
};

bool parse_json(const std::string &s, Json &out) {
    JsonParser jp{s.data(), s.data() + s.size()};
    out = jp.value();
    jp.ws();
    return jp.ok && jp.p == jp.end;
}

// ---- files

bool read_file(const std::string &path, std::string &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    const long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) { fclose(f); return false; }
    out.resize((size_t)n);
    const bool ok = n == 0 || fread(&out[0], 1, (size_t)n, f) == (size_t)n;
    fclose(f);
    return ok;
}

// A git-LFS pointer's object: <git dir>/lfs/objects/<oid[0:2]>/<oid[2:4]>/<oid>, the git dir the nearest .git above the
// file. Returns the path itself for any other file.
std::string resolve_lfs(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || st.st_size >= 1024 || st.st_size == 0) return path;
    std::string s;
    if (!read_file(path, s)) return path;
    const std::string head = "version https://git-lfs.github.com/spec/v1\n";
    if (s.compare(0, head.size(), head) != 0) return path;
    const size_t at = s.find("\noid sha256:");
    if (at == std::string::npos || s.size() < at + 12 + 64) return path;
    const std::string oid = s.substr(at + 12, 64);
    // the git directory: the nearest ancestor (of the file's directory) holding .git
    std::string dir = path;
    const size_t slash = dir.rfind('/');
    dir = slash == std::string::npos ? "." : dir.substr(0, slash);
    if (dir.empty()) dir = "/";
    for (;;) {
        struct stat g;
        const std::string git = dir + (dir == "/" ? "" : "/") + ".git";
        if (stat(git.c_str(), &g) == 0 && S_ISDIR(g.st_mode)) {
            const std::string obj = git + "/lfs/objects/" + oid.substr(0, 2) + "/" + oid.substr(2, 2) + "/" + oid;
            return obj;
        }
        if (dir == "/" || dir == "." || dir.empty()) break;
        const size_t up = dir.rfind('/');
        dir = up == std::string::npos || up == 0 ? "/" : dir.substr(0, up);
    }
    return path;
}

struct Shard {
    std::string path;  // as opened (the object's path for a pointer)
    void *base = nullptr;
    size_t bytes = 0;
    int fd = -1;
};

}  // namespace

struct st_model {
    std::vector<Shard> shards;
    std::vector<std::string> names;
    std::vector<st_tensor_t> tensors;
    std::map<std::string, int> by_name;
};

namespace {

// Maps one file and reads its header into m (tensors appended)
bool add_shard(st_model *m, const std::string &file) {
    const std::string path = resolve_lfs(file);
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) { g_error = "cannot open " + path; return false; }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 8) { close(fd); g_error = path + ": not a safetensors file"; return false; }
    const size_t bytes = (size_t)st.st_size;
    void *base = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { close(fd); g_error = "mmap " + path; return false; }
    const uint8_t *b = (const uint8_t *)base;
    uint64_t hlen = 0;
    for (int i = 7; i >= 0; --i) hlen = (hlen << 8) | b[i];
    if (hlen > bytes - 8) { munmap(base, bytes); close(fd); g_error = path + ": header longer than the file"; return false; }
    std::string hs((const char *)b + 8, (size_t)hlen);
    Json h;
    if (!parse_json(hs, h) || h.kind != Json::Obj) { munmap(base, bytes); close(fd); g_error = path + ": bad header"; return false; }
    const int shard = (int)m->shards.size();
    m->shards.push_back(Shard{path, base, bytes, fd});
    const size_t data0 = 8 + (size_t)hlen;
    for (const auto &kv : h.obj) {
        if (kv.first == "__metadata__") continue;
        const Json &t = kv.second;
        const Json *dt = t.get("dtype"), *sh = t.get("shape"), *off = t.get("data_offsets");
        if (!dt || dt->kind != Json::Str || !sh || sh->kind != Json::Arr || sh->arr.size() > 8 || !off || off->arr.size() != 2) {
            g_error = path + ": bad tensor " + kv.first;
            return false;
        }
        st_tensor_t e;
        memset(&e, 0, sizeof e);
        snprintf(e.dtype, sizeof e.dtype, "%s", dt->str.c_str());
        e.ndim = (int)sh->arr.size();
        for (int i = 0; i < e.ndim; ++i) e.shape[i] = (int64_t)sh->arr[(size_t)i].num;
        const uint64_t b0 = (uint64_t)off->arr[0].num, b1 = (uint64_t)off->arr[1].num;
        if (b1 < b0 || data0 + b1 > bytes) { g_error = path + ": tensor " + kv.first + " outside the file"; return false; }
        e.data = b + data0 + b0;
        e.bytes = (size_t)(b1 - b0);
        e.shard = shard;
        if (m->by_name.count(kv.first)) { g_error = "tensor " + kv.first + " in two files"; return false; }
        m->by_name[kv.first] = (int)m->tensors.size();
        m->names.push_back(kv.first);
        m->tensors.push_back(e);
    }
    return true;
}

}  // namespace

st_model_t *st_open(const char *path) {
    g_error.clear();
    if (!path) { g_error = "st_open: no path"; return nullptr; }
    std::unique_ptr<st_model_t> m(new st_model_t());
    struct stat st;
    if (stat(path, &st) != 0) { g_error = std::string("cannot stat ") + path; return nullptr; }
    if (S_ISDIR(st.st_mode)) {
        const std::string dir = path;
        std::string index;
        if (read_file(dir + "/model.safetensors.index.json", index) && !index.empty()) {
            Json j;
            if (!parse_json(index, j)) { g_error = "bad model.safetensors.index.json"; return nullptr; }
            const Json *wm = j.get("weight_map");
            if (!wm || wm->kind != Json::Obj) { g_error = "index without weight_map"; return nullptr; }
            std::vector<std::string> files;
            for (const auto &kv : wm->obj)
                if (kv.second.kind == Json::Str && std::find(files.begin(), files.end(), kv.second.str) == files.end())
                    files.push_back(kv.second.str);
            for (const auto &f : files)
                if (!add_shard(m.get(), dir + "/" + f)) return nullptr;
        } else if (!add_shard(m.get(), dir + "/model.safetensors")) {
            return nullptr;
        }
    } else if (!add_shard(m.get(), path)) {
        return nullptr;
    }
    return m.release();
}

void st_close(st_model_t *m) {
    if (!m) return;
    for (auto &s : m->shards) {
        if (s.base) munmap(s.base, s.bytes);
        if (s.fd >= 0) close(s.fd);
    }
    delete m;
}

const char *st_last_error(void) { return g_error.c_str(); }

int st_n_tensors(const st_model_t *m) { return m ? (int)m->tensors.size() : 0; }

const char *st_tensor_name(const st_model_t *m, int idx) {
    return m && idx >= 0 && idx < (int)m->names.size() ? m->names[(size_t)idx].c_str() : nullptr;
}

const st_tensor_t *st_find(const st_model_t *m, const char *name) {
    if (!m || !name) return nullptr;
    const auto it = m->by_name.find(name);
    return it == m->by_name.end() ? nullptr : &m->tensors[(size_t)it->second];
}

int st_n_shards(const st_model_t *m) { return m ? (int)m->shards.size() : 0; }

const void *st_shard_base(const st_model_t *m, int shard, size_t *bytes) {
    if (!m || shard < 0 || shard >= (int)m->shards.size()) return nullptr;
    if (bytes) *bytes = m->shards[(size_t)shard].bytes;
    return m->shards[(size_t)shard].base;
}

const char *st_shard_path(const st_model_t *m, int shard) {
    return m && shard >= 0 && shard < (int)m->shards.size() ? m->shards[(size_t)shard].path.c_str() : nullptr;
}

char *st_read_text(const char *dir, const char *name) {
    std::string s;
    if (!dir || !name || !read_file(std::string(dir) + "/" + name, s)) return nullptr;
    char *out = (char *)malloc(s.size() + 1);
    if (!out) return nullptr;
    memcpy(out, s.data(), s.size());
    out[s.size()] = 0;
    return out;
}
