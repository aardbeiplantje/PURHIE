#pragma once
// The image conditioning of the TRELLIS.2 pipeline (get_cond): the image resized to size x size as PIL's LANCZOS does it (8-bit, the
// horizontal pass then the vertical), to [0, 1], ImageNet normalised, through the image encoder (DinoV3): the tokens the flows
// are conditioned on, [tokens(size)][1024] (the CLS token, the 4 registers and the patches).

#include "dinov3.h"

#include <cstdint>
#include <string>
#include <vector>

namespace trellis {

// rgb: w x h x 3 bytes (row-major). cond: [DinoV3::tokens(size)][1024] (the encoder's output, all of its tokens).
bool image_condition(DinoV3& enc, const uint8_t* rgb, int w, int h, int size, std::vector<float>& cond, std::string& error);

// The PIL-style resize of 8-bit RGB (LANCZOS, support 3, the filter stretched by the reduction; each pass rounded to 8 bits).
std::vector<uint8_t> resize_lanczos_rgb8(const uint8_t* rgb, int w, int h, int ow, int oh);
// PIL's resize of 8-bit pixels (ch channels) with a filter: bilinear (support 1), bicubic (a = -0.5, support 2) or Lanczos-3, each
// stretched by the reduction (antialiased), a horizontal then a vertical pass, rounded and clipped to 8 bits.
enum class PilFilter { Bilinear, Bicubic, Lanczos };
std::vector<uint8_t> resize_pil8(const uint8_t* px, int w, int h, int ch, int ow, int oh, PilFilter f);
// The same resize in Pillow's fixed-point arithmetic (its 22-bit taps and integer sums): exact against Pillow's output.
std::vector<uint8_t> resize_pillow8(const uint8_t* px, int w, int h, int ch, int ow, int oh, PilFilter f);

}  // namespace trellis
