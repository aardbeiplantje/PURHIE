// The device chart clustering (src/trellis_charts.cpp compute_charts) against CuMesh's compute_charts on the filled shape mesh (oracle_fill:
// filled_vertices.f32, filled_faces.i32, chart_ids.i32 from CuMesh): the chart count, the faces that differ, and the time.
// Usage: charts_verify [oracle_fill [global_iterations refine_iterations [want.i32 [dump.i32]]]]: the reference defaults to CuMesh's
// chart_ids.i32 (the defaults of CuMesh's compute_charts); a reference from another iteration count (charts_cumesh.py) checks a stage.
#include "trellis_charts.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool read_bytes(const std::string& path, std::vector<char>& v) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    const long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    v.resize((size_t)bytes);
    const size_t got = std::fread(v.data(), 1, v.size(), f);
    std::fclose(f);
    return got == v.size();
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/data/trellis2/oracle_fill";
    const std::string want_path = argc > 4 ? argv[4] : dir + "/chart_ids.i32";
    std::vector<char> vb, fb, cb;
    if (!read_bytes(dir + "/filled_vertices.f32", vb) || !read_bytes(dir + "/filled_faces.i32", fb) || !read_bytes(want_path, cb)) {
        std::fprintf(stderr, "cannot read the oracle in %s (%s)\n", dir.c_str(), want_path.c_str());
        return 1;
    }
    std::vector<float> verts(vb.size() / sizeof(float));
    std::memcpy(verts.data(), vb.data(), vb.size());
    std::vector<int32_t> faces(fb.size() / sizeof(int32_t)), want(cb.size() / sizeof(int32_t));
    std::memcpy(faces.data(), fb.data(), fb.size());
    std::memcpy(want.data(), cb.data(), cb.size());
    if (want.size() != faces.size() / 3) {
        std::fprintf(stderr, "the chart ids are not one per face\n");
        return 1;
    }
    trellis::ChartConfig cfg;  // CuMesh's defaults
    if (argc > 3) {
        cfg.global_iterations = std::atoi(argv[2]);
        cfg.refine_iterations = std::atoi(argv[3]);
    }
    std::vector<int32_t> got;
    trellis::ChartStats stats;
    std::string error;
    if (!trellis::compute_charts(verts, faces, cfg, got, stats, error)) {
        std::fprintf(stderr, "charts: %s\n", error.c_str());
        return 1;
    }
    if (argc > 5) {  // the chart ids for comparison from outside
        FILE* o = std::fopen(argv[5], "wb");
        if (o) {
            std::fwrite(got.data(), sizeof(int32_t), got.size(), o);
            std::fclose(o);
        }
    }
    int want_charts = 0;
    for (int32_t c : want) want_charts = std::max(want_charts, c + 1);
    size_t differ = 0, first = 0;
    for (size_t f = 0; f < got.size(); ++f)
        if (got[f] != want[f]) {
            if (differ == 0) first = f;
            differ++;
        }
    std::printf("charts %d (CuMesh %d), %zu of %zu faces differ%s, %d collapse passes, %.2f s\n", stats.charts, want_charts, differ, got.size(),
                differ ? "" : " (none)", stats.collapse_passes, stats.seconds);
    if (differ) std::printf("first difference at face %zu: %d, CuMesh %d\n", first, got[first], want[first]);
    // the same partition under the renumbering: each of the device's charts is matched to the CuMesh chart of its first face
    std::vector<int> match(stats.charts, -1);
    for (size_t f = 0; f < got.size(); ++f)
        if (got[f] >= 0 && got[f] < stats.charts && match[got[f]] < 0) match[got[f]] = want[f];
    size_t apart = 0;
    for (size_t f = 0; f < got.size(); ++f)
        if (got[f] < 0 || got[f] >= stats.charts || match[got[f]] != want[f]) apart++;
    std::printf("under the renumbering: %zu faces apart (%.2f%%)\n", apart, 100.0 * apart / got.size());
    // CuMesh's own output varies between runs (about 4% of the faces at the defaults, on this mesh): the port passes when the chart count is
    // within 1% of CuMesh's and the partition agrees under the renumbering to within 10%
    const double count_rel = std::fabs((double)stats.charts - want_charts) / want_charts;
    const double apart_frac = (double)apart / got.size();
    const bool ok = count_rel <= 0.01 && apart_frac <= 0.10;
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
