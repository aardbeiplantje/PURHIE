#pragma once
// The pieces of the solid rebuild (trellis_solid.h) that the device kernels (sd_gpu.hip solid_*) and the host reference
// (trellis_solid.cpp) share, so both compute the same numbers: the point-triangle distance and the Kuhn tetrahedra of a cube.

#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
#define SOLID_FN __host__ __device__ inline
#else
#define SOLID_FN inline
#endif

// the squared distance from p to the segment a-b
SOLID_FN float solid_seg_dist2(float px, float py, float pz, float ax, float ay, float az, float bx, float by, float bz) {
    const float ux = bx - ax, uy = by - ay, uz = bz - az;
    const float wx = px - ax, wy = py - ay, wz = pz - az;
    const float uu = ux * ux + uy * uy + uz * uz;
    float t = uu > 0.0f ? (wx * ux + wy * uy + wz * uz) / uu : 0.0f;
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    const float dx = wx - t * ux, dy = wy - t * uy, dz = wz - t * uz;
    return dx * dx + dy * dy + dz * dz;
}

// the squared distance from p to the triangle a, b, c (Ericson, Real-Time Collision Detection 5.1.5: the closest point by the
// Voronoi regions of the vertices, the edges and the face); a degenerate triangle is the nearest of its three edges
SOLID_FN float solid_tri_dist2(float px, float py, float pz, const float* a, const float* b, const float* c) {
    const float abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    const float acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
    const float apx = px - a[0], apy = py - a[1], apz = pz - a[2];
    const float d1 = abx * apx + aby * apy + abz * apz, d2 = acx * apx + acy * apy + acz * apz;
    float qx, qy, qz;
    if (d1 <= 0.0f && d2 <= 0.0f) {
        qx = a[0], qy = a[1], qz = a[2];
    } else {
        const float bpx = px - b[0], bpy = py - b[1], bpz = pz - b[2];
        const float d3 = abx * bpx + aby * bpy + abz * bpz, d4 = acx * bpx + acy * bpy + acz * bpz;
        const float cpx = px - c[0], cpy = py - c[1], cpz = pz - c[2];
        const float d5 = abx * cpx + aby * cpy + abz * cpz, d6 = acx * cpx + acy * cpy + acz * cpz;
        const float vc = d1 * d4 - d3 * d2, vb = d5 * d2 - d1 * d6, va = d3 * d6 - d5 * d4;
        if (d3 >= 0.0f && d4 <= d3) {
            qx = b[0], qy = b[1], qz = b[2];
        } else if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float v = d1 / (d1 - d3);
            qx = a[0] + v * abx, qy = a[1] + v * aby, qz = a[2] + v * abz;
        } else if (d6 >= 0.0f && d5 <= d6) {
            qx = c[0], qy = c[1], qz = c[2];
        } else if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float w = d2 / (d2 - d6);
            qx = a[0] + w * acx, qy = a[1] + w * acy, qz = a[2] + w * acz;
        } else if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            qx = b[0] + w * (c[0] - b[0]), qy = b[1] + w * (c[1] - b[1]), qz = b[2] + w * (c[2] - b[2]);
        } else {
            const float sum = va + vb + vc;
            if (!(sum > 0.0f)) {  // no area: the nearest edge
                const float e0 = solid_seg_dist2(px, py, pz, a[0], a[1], a[2], b[0], b[1], b[2]);
                const float e1 = solid_seg_dist2(px, py, pz, b[0], b[1], b[2], c[0], c[1], c[2]);
                const float e2 = solid_seg_dist2(px, py, pz, c[0], c[1], c[2], a[0], a[1], a[2]);
                return e0 < e1 ? (e0 < e2 ? e0 : e2) : (e1 < e2 ? e1 : e2);
            }
            const float denom = 1.0f / sum;
            const float v = vb * denom, w = vc * denom;
            qx = a[0] + abx * v + acx * w, qy = a[1] + aby * v + acy * w, qz = a[2] + abz * v + acz * w;
        }
    }
    const float dx = px - qx, dy = py - qy, dz = pz - qz;
    return dx * dx + dy * dy + dz * dz;
}

// The six Kuhn tetrahedra of a cube: the corners are dx | dy << 1 | dz << 2, each tetrahedron is a path from corner 0 to corner 7
// along the axes in one order (its permutation), so every edge goes from a corner to one with more bits (a positive direction), and
// a face of the cube is split along the diagonal from its lowest to its highest corner in both cubes that share it. The parity is
// the permutation's: 0 when the corners in this order are positively oriented (det[v1 - v0, v2 - v0, v3 - v0] > 0).
#if defined(__HIPCC__) || defined(__HIP_DEVICE_COMPILE__)
__constant__
#endif
static const int kSolidTets[6][5] = {
    {0, 1, 3, 7, 0}, {0, 1, 5, 7, 1}, {0, 2, 3, 7, 1}, {0, 2, 6, 7, 0}, {0, 4, 5, 7, 0}, {0, 4, 6, 7, 1},
};

// The cell's triangles from its eight corner values fc (negative inside), as corner pairs: each triangle's three vertices are on the
// edges (lo, hi) of the cube's triangulation (lo's bits a subset of hi's), wound with the normal toward the positive side.
// emit(e0lo, e0hi, e1lo, e1hi, e2lo, e2hi) is called for each; returns the count.
template <typename Emit>
SOLID_FN int solid_cell_triangles(const float* fc, Emit emit) {
    int count = 0;
    for (int t = 0; t < 6; ++t) {
        const int* c = kSolidTets[t];
        int o[4], k = 0;
        for (int i = 0; i < 4; ++i)
            if (fc[c[i]] < 0.0f) o[k++] = i;
        const int neg = k;
        if (neg == 0 || neg == 4) continue;
        for (int i = 0; i < 4; ++i)
            if (!(fc[c[i]] < 0.0f)) o[k++] = i;
        int inv = c[4];
        for (int i = 0; i < 4; ++i)
            for (int j = i + 1; j < 4; ++j) inv += o[i] > o[j];
        const bool flip = inv & 1;  // (a, b, c, d) = the reordered corners are negatively oriented: reverse the windings
        const int a = c[o[0]], b = c[o[1]], cc = c[o[2]], d = c[o[3]];
        // an edge as (lo, hi) by the corners' order along the path (a corner's bits contain the earlier ones')
#define SOLID_E(p, q) ((p) < (q) ? (p) : (q)), ((p) < (q) ? (q) : (p))
        if (neg == 1) {  // a inside: the face away from a
            if (!flip) emit(SOLID_E(a, b), SOLID_E(a, cc), SOLID_E(a, d));
            else emit(SOLID_E(a, b), SOLID_E(a, d), SOLID_E(a, cc));
            count += 1;
        } else if (neg == 3) {  // d outside: the face toward d
            if (!flip) emit(SOLID_E(d, a), SOLID_E(d, b), SOLID_E(d, cc));
            else emit(SOLID_E(d, a), SOLID_E(d, cc), SOLID_E(d, b));
            count += 1;
        } else {  // a, b inside, cc, d outside: the quad ac, ad, bd, bc
            if (!flip) {
                emit(SOLID_E(a, cc), SOLID_E(a, d), SOLID_E(b, d));
                emit(SOLID_E(a, cc), SOLID_E(b, d), SOLID_E(b, cc));
            } else {
                emit(SOLID_E(a, cc), SOLID_E(b, d), SOLID_E(a, d));
                emit(SOLID_E(a, cc), SOLID_E(b, cc), SOLID_E(b, d));
            }
            count += 2;
        }
#undef SOLID_E
    }
    return count;
}
