/* safetensors_mmap.h -- a checkpoint read in place: a directory (model.safetensors.index.json, or a single
 * model.safetensors) or one shard, its files mapped read-only, the tensors addressed by name. Files that are git-LFS
 * pointers (a clone without checkout) are resolved to their objects in the git directory (<git>/lfs/objects/xx/yy/oid),
 * as the converter does. No copies: the data pointers are the mappings' (registered for the device by the caller). */
#ifndef SAFETENSORS_MMAP_H
#define SAFETENSORS_MMAP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_model st_model_t;

typedef struct {
    const void *data;   /* the tensor's bytes in the mapping */
    size_t bytes;
    char dtype[8];      /* "BF16", "F32", ... (the header's string) */
    int ndim;
    int64_t shape[8];   /* the checkpoint's order (row-major: shape[ndim - 1] is the fastest) */
    int shard;          /* the index of the file (st_shard_base) */
} st_tensor_t;

/* path: a directory or a file. Returns NULL on failure (st_last_error). */
st_model_t *st_open(const char *path);
void st_close(st_model_t *m);
const char *st_last_error(void);

int st_n_tensors(const st_model_t *m);
const char *st_tensor_name(const st_model_t *m, int idx);
/* the tensor by name, or NULL */
const st_tensor_t *st_find(const st_model_t *m, const char *name);

/* The mappings (one per file): base and size, for the device registration of the whole file */
int st_n_shards(const st_model_t *m);
const void *st_shard_base(const st_model_t *m, int shard, size_t *bytes);
/* the file's path as opened (the object's path for an LFS pointer) */
const char *st_shard_path(const st_model_t *m, int shard);

/* a file's text (config.json, pdd_config.json): malloc'd, NUL-terminated; free with free(). NULL on failure. */
char *st_read_text(const char *dir, const char *name);

/* JSON text: the first number after "key": (0 if not found); the numbers of the array after "key": [ ... ] (up to max; the
 * count returned, -1 if the key is missing). Plain scans: the configs' own keys are unique enough for these. */
int st_json_number(const char *text, const char *key, double *out);
int st_json_numbers(const char *text, const char *key, double *out, int max);

#ifdef __cplusplus
}
#endif

#endif /* SAFETENSORS_MMAP_H */
