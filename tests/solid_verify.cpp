// The solid rebuild for 3D printing (src/trellis_solid.h) and the simplifier's manifold mode.
//
//   solid_verify                                   synthetic meshes: a cube with half its faces turned, a cube with a small hole and one
//                                                  with a large one, a cube with an inner wall, two cubes sharing an edge, a speck; each
//                                                  surface closed, wound outward, with the volume it should have
//   solid_verify MESH [OUT.stl] [GRID] [FACES] [CLOSE]  a real mesh (a binary STL, or the stem of a .f32/.i32 pair: MESH_vertices.f32 and
//                                                  MESH_faces.i32, or a directory with mesh_vertices.f32 and mesh_faces.i32): the device
//                                                  against the host reference on a 96 grid (the grids and the surface), then the device
//                                                  at GRID (512) with the simplification to FACES (700000), both checked; OUT the result
#include "trellis_simplify.h"
#include "trellis_solid.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <sys/stat.h>
#include <tuple>
#include <vector>

using namespace trellis;

static int g_fail = 0;
#define CHECK(cond, ...)                 \
    do {                                 \
        if (!(cond)) {                   \
            std::printf("FAIL: ");       \
            std::printf(__VA_ARGS__);    \
            std::printf("\n");           \
            ++g_fail;                    \
        }                                \
    } while (0)

// ---- meshes

struct Mesh {
    std::vector<float> v;
    std::vector<int32_t> f;
};

// the surface of the box [lo, hi]^3 in n x n quads a side, wound outward
static void add_box(Mesh& m, const float lo[3], const float hi[3], int n) {
    for (int axis = 0; axis < 3; ++axis)
        for (int side = 0; side < 2; ++side) {
            const int u = (axis + 1) % 3, w = (axis + 2) % 3;
            const int base = (int)(m.v.size() / 3);
            for (int j = 0; j <= n; ++j)
                for (int i = 0; i <= n; ++i) {
                    float p[3];
                    p[axis] = side ? hi[axis] : lo[axis];
                    p[u] = lo[u] + (hi[u] - lo[u]) * i / n;
                    p[w] = lo[w] + (hi[w] - lo[w]) * j / n;
                    m.v.insert(m.v.end(), p, p + 3);
                }
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i) {
                    const int a = base + j * (n + 1) + i, b = a + 1, c = a + n + 1, d = c + 1;
                    // (u, w, axis) is right-handed: (b - a) x (c - a) points along +axis
                    if (side) m.f.insert(m.f.end(), {a, b, d, a, d, c});
                    else m.f.insert(m.f.end(), {a, d, b, a, c, d});
                }
        }
}

static double box_volume(const float lo[3], const float hi[3], float grow) {
    return (double)(hi[0] - lo[0] + 2 * grow) * (hi[1] - lo[1] + 2 * grow) * (hi[2] - lo[2] + 2 * grow);
}

static bool read_file(const std::string& path, std::vector<char>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::fseek(fp, 0, SEEK_END);
    const long n = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    out.resize(n > 0 ? (size_t)n : 0);
    const bool ok = n >= 0 && std::fread(out.data(), 1, out.size(), fp) == out.size();
    std::fclose(fp);
    return ok;
}

// a binary STL, its vertices welded by their exact positions
static bool read_stl(const std::string& path, Mesh& m) {
    std::vector<char> b;
    if (!read_file(path, b) || b.size() < 84) return false;
    uint32_t n;
    std::memcpy(&n, b.data() + 80, 4);
    if (b.size() != 84 + 50 * (size_t)n) return false;
    std::map<std::tuple<float, float, float>, int32_t> at;
    for (uint32_t t = 0; t < n; ++t) {
        const char* p = b.data() + 84 + 50 * (size_t)t + 12;
        for (int k = 0; k < 3; ++k) {
            float x[3];
            std::memcpy(x, p + 12 * k, 12);
            const auto key = std::make_tuple(x[0], x[1], x[2]);
            auto it = at.find(key);
            if (it == at.end()) {
                it = at.emplace(key, (int32_t)(m.v.size() / 3)).first;
                m.v.insert(m.v.end(), x, x + 3);
            }
            m.f.push_back(it->second);
        }
    }
    return true;
}

static bool read_pair(const std::string& vpath, const std::string& fpath, Mesh& m) {
    std::vector<char> a, b;
    if (!read_file(vpath, a) || !read_file(fpath, b) || a.size() % 12 || b.size() % 12) return false;
    m.v.resize(a.size() / 4);
    m.f.resize(b.size() / 4);
    std::memcpy(m.v.data(), a.data(), a.size());
    std::memcpy(m.f.data(), b.data(), b.size());
    return true;
}

static bool read_mesh(const std::string& path, Mesh& m) {
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) return read_pair(path + "/mesh_vertices.f32", path + "/mesh_faces.i32", m);
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".stl") == 0) return read_stl(path, m);
    return read_pair(path + "_vertices.f32", path + "_faces.i32", m);
}

static bool write_stl(const std::string& path, const Mesh& m) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    char head[80] = "solid_verify";
    std::fwrite(head, 1, 80, fp);
    const uint32_t n = (uint32_t)(m.f.size() / 3);
    std::fwrite(&n, 4, 1, fp);
    for (uint32_t t = 0; t < n; ++t) {
        const float* a = &m.v[3 * m.f[3 * t]];
        const float* b = &m.v[3 * m.f[3 * t + 1]];
        const float* c = &m.v[3 * m.f[3 * t + 2]];
        float nr[3] = {(b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]), (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]),
                       (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])};
        const float l = std::sqrt(nr[0] * nr[0] + nr[1] * nr[1] + nr[2] * nr[2]);
        for (float& x : nr) x = l > 0 ? x / l : 0.0f;
        const uint16_t attr = 0;
        std::fwrite(nr, 4, 3, fp);
        std::fwrite(a, 4, 3, fp);
        std::fwrite(b, 4, 3, fp);
        std::fwrite(c, 4, 3, fp);
        std::fwrite(&attr, 2, 1, fp);
    }
    return std::fclose(fp) == 0;
}

static void print_check(const char* what, const Mesh& m) {
    MeshCheck c;
    check_mesh(m.v, m.f, c);
    std::printf("  %s: %zu faces, %zu vertices; edges %ld: boundary %ld, non-manifold %ld, misoriented %ld; degenerate faces %ld; "
                "pieces %d; volume %.6g\n",
                what, m.f.size() / 3, m.v.size() / 3, c.edges, c.boundary, c.nonmanifold, c.misoriented, c.degenerate, c.pieces, c.volume);
}

// ---- the synthetic cases: a closed outward surface, one piece per expected solid, the volume of the boxes grown by iso cells

static void synthetic() {
    SolidConfig cfg;
    cfg.grid = 128;
    struct Case {
        const char* name;
        Mesh m;
        int pieces;
        double volume;     // expected (0: not checked)
        double tolerance;  // relative
        float wall = 0, drain = 0;  // a hollow
        int hollows = 0, drains = 0;
        int cavities = -1;  // the model's own sealed hollows kept (-1: not checked)
    };
    std::vector<Case> cases;
    const float lo[3] = {-0.5f, -0.4f, -0.3f}, hi[3] = {0.5f, 0.4f, 0.3f};
    const float h = 1.0f / cfg.grid, grow = cfg.iso * h;
    {
        Case c{"a box, half its faces turned", {}, 1, box_volume(lo, hi, grow), 0.01};
        c.cavities = 0;  // its sealed inside is a solid's (one wall to the outside): filled
        add_box(c.m, lo, hi, 24);
        std::mt19937 rng(1);
        for (size_t t = 0; t < c.m.f.size() / 3; ++t)
            if (rng() & 1) std::swap(c.m.f[3 * t + 1], c.m.f[3 * t + 2]);
        cases.push_back(std::move(c));
    }
    {
        // a hole of one quad (1/24 of the side: 5.3 cells at 128, under twice close): closed by the plug, the box solid
        Case c{"a box with a small hole", {}, 1, box_volume(lo, hi, grow), 0.01};
        add_box(c.m, lo, hi, 24);
        c.m.f.erase(c.m.f.begin() + 6 * (24 * 12 + 12), c.m.f.begin() + 6 * (24 * 12 + 12) + 6);
        cases.push_back(std::move(c));
    }
    {
        // a hole of 12 x 12 quads (51 x 38 cells, wider than twice close_max): the fill goes in at every close, the wall becomes a
        // closed shell 2 iso thick, kept from the first run
        Case c{"a box with a large hole (a thin shell)", {}, 1, 0, 0};
        add_box(c.m, lo, hi, 24);
        std::vector<int32_t> keep;
        for (int j = 0; j < 24; ++j)
            for (int i = 0; i < 24; ++i)
                if (!(i >= 6 && i < 18 && j >= 6 && j < 18)) keep.insert(keep.end(), c.m.f.begin() + 6 * (24 * j + i), c.m.f.begin() + 6 * (24 * j + i) + 6);
        keep.insert(keep.end(), c.m.f.begin() + 6 * 24 * 24, c.m.f.end());
        c.m.f = keep;
        cases.push_back(std::move(c));
    }
    {
        // an inner wall (x = 0) inside the box: no surface inside the solid
        Case c{"a box with an inner wall", {}, 1, box_volume(lo, hi, grow), 0.01};
        add_box(c.m, lo, hi, 24);
        const float wl[3] = {0.0f, lo[1], lo[2]}, wh[3] = {0.0f, hi[1], hi[2]};
        Mesh wall;
        add_box(wall, wl, wh, 8);  // a flat box: its two x faces lie on x = 0
        const int base = (int)(c.m.v.size() / 3);
        c.m.v.insert(c.m.v.end(), wall.v.begin(), wall.v.end());
        for (int32_t x : wall.f) c.m.f.push_back(base + x);
        cases.push_back(std::move(c));
    }
    {
        // two boxes sharing an edge (non-manifold there): one solid
        Case c{"two boxes sharing an edge", {}, 1, 0, 0};
        const float a0[3] = {-0.5f, -0.5f, -0.2f}, a1[3] = {0.0f, 0.0f, 0.2f}, b0[3] = {0.0f, 0.0f, -0.2f}, b1[3] = {0.5f, 0.5f, 0.2f};
        add_box(c.m, a0, a1, 8);
        add_box(c.m, b0, b1, 8);
        cases.push_back(std::move(c));
    }
    {
        // a box and a speck far from it (a tiny box, under min_piece of the volume): the speck is dropped
        Case c{"a box and a speck", {}, 1, 0, 0};
        const float s0[3] = {0.45f, 0.45f, 0.45f}, s1[3] = {0.46f, 0.46f, 0.46f};
        add_box(c.m, lo, hi, 8);
        add_box(c.m, s0, s1, 1);
        cases.push_back(std::move(c));
    }
    {
        // the model's own sealed hollow (a box inside a box, 0.1 apart: two walls to the outside): kept, as air. The outer
        // surface iso out, the hollow's iso in from the inner box
        Case c{"a box with a sealed hollow of its own", {}, 2, 0, 0.02};
        add_box(c.m, lo, hi, 12);
        const float ilo[3] = {lo[0] + 0.1f, lo[1] + 0.1f, lo[2] + 0.1f}, ihi[3] = {hi[0] - 0.1f, hi[1] - 0.1f, hi[2] - 0.1f};
        add_box(c.m, ilo, ihi, 12);
        c.cavities = 1;
        double in = 1;
        for (int k = 0; k < 3; ++k) in *= ihi[k] - ilo[k] - 2 * grow;
        c.volume = box_volume(lo, hi, grow) - in;
        cases.push_back(std::move(c));
    }
    {
        // hollow, walls of 0.1 (12.8 cells), no drain: the outer surface and the hollow's (two pieces); the volume the box grown by
        // iso less the hollow, a box 0.1 in from that
        Case c{"a hollow box", {}, 2, 0, 0.05};
        add_box(c.m, lo, hi, 8);
        c.wall = 0.1f;
        c.hollows = 1;
        double in = 1;
        for (int k = 0; k < 3; ++k) in *= hi[k] - lo[k] + 2 * grow - 2 * c.wall;
        c.volume = box_volume(lo, hi, grow) - in;
        cases.push_back(std::move(c));
    }
    {
        // the same with a drain hole 0.04 wide: one surface (the hollow open to the outside through the bottom)
        Case c{"a hollow box with a drain", {}, 1, 0, 0.05};
        add_box(c.m, lo, hi, 8);
        c.wall = 0.1f;
        c.drain = 0.04f;
        c.hollows = 1;
        c.drains = 1;
        double in = 1;
        for (int k = 0; k < 3; ++k) in *= hi[k] - lo[k] + 2 * grow - 2 * c.wall;
        c.volume = box_volume(lo, hi, grow) - in;
        cases.push_back(std::move(c));
    }
    for (Case& c : cases) {
        std::printf("%s (%zu faces in)\n", c.name, c.m.f.size() / 3);
        Mesh out, ref;
        SolidStats st, rst;
        std::string err;
        SolidConfig ccfg = cfg;
        ccfg.wall = c.wall;
        ccfg.drain = c.drain;
        const bool ok = solidify(c.m.v, c.m.f, ccfg, out.v, out.f, st, err);
        CHECK(ok, "%s: %s", c.name, err.c_str());
        if (!ok) continue;
        const bool rok = solidify_reference(c.m.v, c.m.f, ccfg, ref.v, ref.f, rst, err);
        CHECK(rok, "%s: the reference: %s", c.name, err.c_str());
        MeshCheck mc;
        check_mesh(out.v, out.f, mc);
        print_check("the solid", out);
        std::printf("  grid %d x %d x %d, %d fill rounds, %d pieces (%d dropped); close %g after %d runs, mean thickness %.1f cells; %.2f s\n",
                    st.dims[0], st.dims[1], st.dims[2], st.flood_rounds, st.pieces, st.pieces_dropped, st.close, st.attempts, st.thickness,
                    st.seconds);
        const bool thin = std::strstr(c.name, "large hole") != nullptr;
        CHECK(thin ? (st.attempts == 3 && st.close == cfg.close) : (st.attempts == 1 + (c.wall > 0) && st.close == cfg.close),
              "%s: close %g after %d runs", c.name, st.close, st.attempts);
        std::printf("  sealed regions: %d kept as the model's hollows, %d filled\n", st.cavities, st.interiors);
        if (c.cavities >= 0) CHECK(st.cavities == c.cavities, "%s: %d sealed hollows kept, %d expected", c.name, st.cavities, c.cavities);
        if (c.wall > 0) {
            std::printf("  %d hollows (%d filled), %d drains\n", st.hollows, st.hollows_filled, st.drains);
            CHECK(st.hollows - st.hollows_filled == c.hollows && st.drains == c.drains, "%s: %d hollows (%d filled) and %d drains", c.name,
                  st.hollows, st.hollows_filled, st.drains);
        }
        CHECK(mc.closed(), "%s: not closed (boundary %ld, non-manifold %ld, misoriented %ld)", c.name, mc.boundary, mc.nonmanifold, mc.misoriented);
        CHECK(mc.volume > 0, "%s: wound inward (volume %g)", c.name, mc.volume);
        CHECK(mc.pieces == c.pieces, "%s: %d pieces, %d expected", c.name, mc.pieces, c.pieces);
        if (c.volume > 0)
            CHECK(std::fabs(mc.volume - c.volume) <= c.tolerance * c.volume, "%s: volume %g, %g expected", c.name, mc.volume, c.volume);
        if (std::strstr(c.name, "large hole"))
            CHECK(mc.volume < 0.2 * box_volume(lo, hi, 0), "%s: volume %g, a thin shell expected", c.name, mc.volume);
        if (rok) {
            CHECK(ref.f.size() == out.f.size() && ref.v.size() == out.v.size(), "%s: the reference has %zu faces and %zu vertices, the device %zu and %zu",
                  c.name, ref.f.size() / 3, ref.v.size() / 3, out.f.size() / 3, out.v.size() / 3);
            if (ref.f == out.f && ref.v.size() == out.v.size()) {
                float dmax = 0;
                for (size_t i = 0; i < ref.v.size(); ++i) dmax = std::fmax(dmax, std::fabs(ref.v[i] - out.v[i]));
                std::printf("  the reference: the same faces, vertices within %.3g\n", dmax);
                CHECK(dmax < 1e-4f * 1.0f, "%s: the reference's vertices differ by %g", c.name, dmax);
            } else if (ref.f != out.f) {
                std::printf("  the reference: other faces\n");
            }
        }
        // the simplification keeps it closed
        Mesh s = out;
        SimplifyStats sst;
        const int target = (int)(out.f.size() / 3 / 8);
        const bool sok = simplify_to_faces(s.v, s.f, target, sst, err, true);
        CHECK(sok, "%s: simplify: %s", c.name, err.c_str());
        if (sok) {
            MeshCheck sc;
            check_mesh(s.v, s.f, sc);
            print_check("simplified", s);
            CHECK(sc.closed(), "%s: the simplified surface is not closed (boundary %ld, non-manifold %ld, misoriented %ld)", c.name, sc.boundary,
                  sc.nonmanifold, sc.misoriented);
            CHECK(sc.pieces == mc.pieces, "%s: the simplification made %d pieces of %d", c.name, sc.pieces, mc.pieces);
            CHECK(std::fabs(sc.volume - mc.volume) < 0.02 * std::fabs(mc.volume), "%s: the simplification changed the volume %g to %g", c.name,
                  mc.volume, sc.volume);
        }
    }
}

// ---- a real mesh

static void compare_grids(const Mesh& m) {
    SolidConfig cfg;
    cfg.grid = 96;
    SolidGrids dg, rg;
    std::string err;
    Mesh ref;
    SolidStats rst;
    const bool ok = solidify_grids(m.v, m.f, cfg, dg, err);
    CHECK(ok, "the device grids: %s", err.c_str());
    const bool rok = solidify_reference(m.v, m.f, cfg, ref.v, ref.f, rst, err, &rg);
    CHECK(rok, "the reference: %s", err.c_str());
    if (!ok || !rok) return;
    const size_t n = rg.udf.size();
    CHECK(dg.udf.size() == n, "the grids' sizes differ");
    if (dg.udf.size() != n) return;
    float du = 0, dfield = 0;
    long inf_diff = 0, d1_diff = 0, st_diff = 0;
    for (size_t i = 0; i < n; ++i) {
        if (std::isinf(rg.udf[i]) != std::isinf(dg.udf[i])) ++inf_diff;
        else if (!std::isinf(rg.udf[i])) du = std::fmax(du, std::fabs(rg.udf[i] - dg.udf[i]));
        d1_diff += rg.d1[i] != dg.d1[i];
        st_diff += rg.state[i] != dg.state[i];
        dfield = std::fmax(dfield, std::fabs(rg.field[i] - dg.field[i]));
    }
    std::printf("device and reference at grid %d (%d x %d x %d): the distance within %.3g cells (%ld differ in the band), the wall's "
                "transform %ld differ, the fill %ld differ, the field within %.3g\n",
                cfg.grid, rst.dims[0], rst.dims[1], rst.dims[2], du, inf_diff, d1_diff, st_diff, dfield);
    CHECK(du < 1e-4f, "the distance to the mesh differs by %g cells", du);
    CHECK(inf_diff <= (long)(n / 100000), "%ld points in the band of one only", inf_diff);
    CHECK(st_diff <= (long)(n / 100000), "%ld points filled differently", st_diff);
    CHECK(dfield < 1e-3f || st_diff + d1_diff > 0, "the field differs by %g", dfield);
    Mesh out;
    SolidStats st;
    const bool sok = solidify(m.v, m.f, cfg, out.v, out.f, st, err);
    CHECK(sok, "the device surface: %s", err.c_str());
    if (sok) {
        std::printf("  surfaces: the device %zu faces, the reference %zu\n", out.f.size() / 3, ref.f.size() / 3);
        CHECK(std::labs((long)out.f.size() - (long)ref.f.size()) <= (long)(ref.f.size() / 1000), "the surfaces' face counts differ");
    }
}

static void real(const std::string& path, const std::string& out_path, int grid, int faces, float close) {
    Mesh m;
    if (!read_mesh(path, m)) {
        std::printf("FAIL: can not read %s\n", path.c_str());
        ++g_fail;
        return;
    }
    std::printf("%s\n", path.c_str());
    print_check("the mesh", m);
    compare_grids(m);
    SolidConfig cfg;
    cfg.grid = grid;
    if (close > 0) cfg.close = close;
    Mesh out;
    SolidStats st;
    std::string err;
    if (!solidify(m.v, m.f, cfg, out.v, out.f, st, err)) {
        std::printf("FAIL: solidify: %s\n", err.c_str());
        ++g_fail;
        return;
    }
    std::printf("grid %d: %d x %d x %d points, cell %.4g; %d fill rounds, %ld outside; %d pieces, %d dropped; volume %.6g; close %g after "
                "%d runs, mean thickness %.1f cells; sealed regions: %d kept as hollows, %d filled\n", grid, st.dims[0], st.dims[1], st.dims[2],
                st.cell, st.flood_rounds, st.outside, st.pieces, st.pieces_dropped, st.volume, st.close, st.attempts, st.thickness, st.cavities,
                st.interiors);
    std::printf("  seconds: distance %.2f, transform %.2f, fill %.2f, surface %.2f; all %.2f\n", st.seconds_distance, st.seconds_transform,
                st.seconds_fill, st.seconds_surface, st.seconds);
    MeshCheck c;
    check_mesh(out.v, out.f, c);
    print_check("the solid", out);
    CHECK(c.closed(), "the solid is not closed");
    CHECK(c.volume > 0, "the solid is wound inward");
    SimplifyStats sst;
    const auto t0 = std::chrono::steady_clock::now();
    if (!simplify_to_faces(out.v, out.f, faces, sst, err, true)) {
        std::printf("FAIL: simplify: %s\n", err.c_str());
        ++g_fail;
        return;
    }
    std::printf("simplified to %d faces in %d passes, %.2f s\n", sst.faces_after,  sst.passes,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    MeshCheck s;
    check_mesh(out.v, out.f, s);
    print_check("simplified", out);
    CHECK(s.closed(), "the simplified solid is not closed");
    CHECK(s.pieces == c.pieces, "the simplification made %d pieces of %d", s.pieces, c.pieces);
    if (!out_path.empty()) {
        CHECK(write_stl(out_path, out), "can not write %s", out_path.c_str());
        std::printf("wrote %s\n", out_path.c_str());
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IOLBF, 0);
    if (argc < 2) synthetic();
    else real(argv[1], argc > 2 ? argv[2] : "", argc > 3 ? std::atoi(argv[3]) : 512, argc > 4 ? std::atoi(argv[4]) : 700000,
              argc > 5 ? (float)std::atof(argv[5]) : 0.0f);
    std::printf(g_fail ? "%d FAILED\n" : "all passed\n", g_fail);
    return g_fail ? 1 : 0;
}
