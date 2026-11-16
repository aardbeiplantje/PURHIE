// hip_check.h: the HIP runtime's results are checked (hipError_t is [[nodiscard]]: a discarded result is a warning).
// HIP_CHECK(call): on failure, throws std::runtime_error naming the call and HIP's message.
// HIP_WARN(call): on failure, prints the same to stderr and goes on: for cleanup that must not throw (destructors,
// the error path of a graph capture).
#ifndef HIP_CHECK_H
#define HIP_CHECK_H

#include <hip/hip_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#define HIP_CHECK(call)                                                                   \
    do {                                                                                  \
        const hipError_t hip_check_e_ = (call);                                           \
        if (hip_check_e_ != hipSuccess)                                                   \
            throw std::runtime_error(std::string(#call) + " failed: " + hipGetErrorString(hip_check_e_)); \
    } while (0)

#define HIP_WARN(call)                                                                    \
    do {                                                                                  \
        const hipError_t hip_warn_e_ = (call);                                            \
        if (hip_warn_e_ != hipSuccess)                                                    \
            std::fprintf(stderr, "%s failed: %s\n", #call, hipGetErrorString(hip_warn_e_)); \
    } while (0)

#endif  // HIP_CHECK_H
