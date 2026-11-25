#pragma once
// CuMesh's chart clustering (cumesh-ssh/src/atlas.cu compute_charts) on the device, for the UV unwrap: the faces of a mesh are grouped into
// charts by cone-angle clustering. Each global iteration collapses the chart graph (the cheapest edges whose merged cone stays under the
// threshold, with area and perimeter penalties), refines the faces onto the neighbouring charts whose cones hold their normals, and
// splits the charts that became disconnected. The per-face, per-edge and per-chart steps are sd_gpu.hip chart_* kernels; the sorts of
// the chart pairs and the chart-face lists are on the host, in CuMesh's order (stable sorts, ids ranked by the first face).

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

struct ChartConfig {
    float threshold_cone_half_angle_rad = 1.5707963267948966f;  // radians(90), CuMesh's default
    int refine_iterations = 100;
    int global_iterations = 3;
    float smooth_strength = 1.0f;
    float area_penalty_weight = 0.1f;
    float perimeter_area_ratio_weight = 1e-4f;
};

struct ChartStats {
    int charts = 0;
    int collapse_passes = 0;  // the collapse steps over all global iterations
    double seconds = 0;
};

// verts [V][3], faces [F][3] (no degenerate faces: python's remove_degenerate_faces runs first): chart_ids [F], the chart of each face,
// numbered 0..charts-1 in the order of each chart's smallest face (CuMesh's numbering). Needs the device.
bool compute_charts(const std::vector<float>& verts, const std::vector<int32_t>& faces, const ChartConfig& cfg, std::vector<int32_t>& chart_ids,
                    ChartStats& stats, std::string& error);

}  // namespace trellis
