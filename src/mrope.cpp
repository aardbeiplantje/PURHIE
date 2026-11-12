#include "mrope.h"

#include <algorithm>

int mrope_positions(const int32_t* ids, int n, const int32_t* grid_hw, int n_images, int merge, int32_t image_token,
                    int64_t* pos_out, int64_t* delta_out) {
    if (!ids || !pos_out || n <= 0 || merge <= 0 || n_images < 0) return -1;
    int64_t cur = 0;  // the next text position
    int img = 0;
    int i = 0;
    while (i < n) {
        if (ids[i] == image_token) {
            // a run of pad tokens: one image (its grid gives the run's length)
            if (img >= n_images) return -1;
            const int gh = grid_hw[2 * img] / merge, gw = grid_hw[2 * img + 1] / merge;
            const int run = gh * gw;
            int j = i;
            while (j < n && ids[j] == image_token) ++j;
            if (j - i != run) return -1;
            for (int r = 0; r < gh; ++r)
                for (int c = 0; c < gw; ++c) {
                    const int k = i + r * gw + c;
                    pos_out[3 * k + 0] = cur;
                    pos_out[3 * k + 1] = cur + r;
                    pos_out[3 * k + 2] = cur + c;
                }
            cur += std::max(gh, gw);
            ++img;
            i = j;
        } else {
            pos_out[3 * i + 0] = pos_out[3 * i + 1] = pos_out[3 * i + 2] = cur;
            ++cur;
            ++i;
        }
    }
    if (img != n_images) return -1;
    if (delta_out) {
        int64_t mx = 0;
        for (int k = 0; k < 3 * n; ++k) mx = std::max(mx, pos_out[k]);
        *delta_out = mx + 1 - n;
    }
    return 0;
}
