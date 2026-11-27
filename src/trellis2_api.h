/* TRELLIS.2, an image to a textured 3D model (binary glTF), for the server: the C interface of trellis_image_to_3d.h with one job at a
 * time on a background thread, so the caller's loop keeps serving while a job runs (minutes). The job holds the device's run mutex
 * (sd_device.h) while it runs: the other models of the device layer (the image and audio generators) must not run meanwhile. */
#ifndef TRELLIS2_API_H
#define TRELLIS2_API_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct t2_pipe t2_pipe_t;

/* Opens the models: the TRELLIS.2-4B directory (pipeline.json, ckpts/), TRELLIS-image-large's sparse-structure decoder
 * (ss_dec_conv3d_16l8_fp16.safetensors), DINOv3 (dino_v3_L_naf_fp32.safetensors) and BiRefNet (birefnet.safetensors; NULL or "": none,
 * then only images with an alpha channel are cut out). NULL on error (t2_last_error). */
t2_pipe_t* t2_open(const char* trellis2_dir, const char* ss_dec, const char* dinov3, const char* birefnet);
/* Waits for a running job, then releases the models. */
void t2_close(t2_pipe_t* p);
const char* t2_last_error(void);
/* 1 when the pipeline removes backgrounds (BiRefNet), else 0 */
int t2_has_background_removal(t2_pipe_t* p);

/* Starts a job: an encoded image (PNG, JPEG, BMP, GIF or TGA: stb_image), the resolution (512, or 1024: the cascade), the seed, the
 * faces after the simplification, the texture size (texels per side; 0: no texture, the mesh alone, which is faster), remove_background
 * (-1: the alpha when the image has one, else BiRefNet; 0: the whole image; 1: BiRefNet), the format (0: a binary glTF; 1: a binary STL
 * of the mesh, always without texture, for 3D printing) and the STL's largest side in millimetres (0: the mesh's own units).
 * Without texture (the mesh alone, the STL) the mesh is rebuilt as a closed solid first (trellis_solid.h). hollow_mm (an STL with a
 * size): hollow, the walls this thick, at least 2 (0: solid); drain_mm: each hollow's drain hole, this wide, from its lowest point down (0: none).
 * 0: started; 1: a job is running (none started); -1: an error (t2_last_error: the image, an option, or no background removal for an
 * image without alpha). */
int t2_start(t2_pipe_t* p, const unsigned char* image, size_t n, int resolution, unsigned long long seed, int faces, int texture_size,
             int remove_background, int format, double stl_size_mm, double hollow_mm, double drain_mm);
/* The job: 0 running, 1 done (its GLB waits for t2_take), -1 failed (t2_take gives its error), 2 no job. stage: copied (at most
 * stage_len bytes with the NUL); step / steps: the stage's (0, 0 when it has none); seconds: since the start (to the end when done). */
int t2_poll(t2_pipe_t* p, char* stage, size_t stage_len, int* step, int* steps, double* seconds);
/* The finished job's model, GLB or STL (*glb malloc'ed: t2_free), and the job is cleared. 0; -1 when it failed (t2_last_error: its error, and the
 * job is cleared) or no job is done. */
/* Asks the running job to stop: it ends at its next progress report (a sampler's step, a stage) as a failed job with the error
 * "cancelled". 1 when a job was running (the request is made), 0 when none is (or no pipeline). */
int t2_cancel(t2_pipe_t* p);
int t2_take(t2_pipe_t* p, unsigned char** glb, size_t* n);
void t2_free(void* ptr);

#ifdef __cplusplus
}
#endif
#endif /* TRELLIS2_API_H */
