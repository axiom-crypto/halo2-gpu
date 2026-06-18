// borrowed from https://github.com/supranational/sppark/blob/main/util/exception.cuh
// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

class cuda_error : public std::runtime_error {
    cudaError_t _code;

public:
    cuda_error(cudaError_t err, const std::string& reason)
        : std::runtime_error { reason }
    {
        _code = err;
    }
    inline cudaError_t code() const
    {
        return _code;
    }
};

template <typename... Types>
inline std::string fmt(const char* fmt, Types... args)
{
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string ret(++len, '\0');
    std::snprintf(&ret.front(), len, fmt, args...);
    ret.resize(--len);
    return ret;
}

#define CUDA_OK(expr)                                                        \
    do {                                                                     \
        cudaError_t code = expr;                                             \
        if (code != cudaSuccess) {                                           \
            auto str = fmt("%s@%s:%d failed: %s", #expr, __FILE__, __LINE__, \
                cudaGetErrorString(code));                                   \
            throw cuda_error(code, str);                                     \
        }                                                                    \
    } while (0)

class cpp_error : public std::runtime_error {
    uint32_t _code;

public:
    cpp_error(const std::string& reason)
        : std::runtime_error { reason }
    {
        _code = 0xffffffff; // hardcode for cpp error
    }
    inline uint32_t code() const
    {
        return _code;
    }
};

#define CPP_CHECK(expr, message)                                               \
    do {                                                                       \
        bool res = expr;                                                       \
        if (res == false) {                                                    \
            auto str = fmt("%s @ %s:%d failed: %s", #expr, __FILE__, __LINE__, \
                message);                                                      \
            printf("%s\r\n", str.c_str());                                     \
            throw cpp_error(str);                                              \
        }                                                                      \
    } while (0)
