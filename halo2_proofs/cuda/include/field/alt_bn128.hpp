// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Device-side BN254 fr_t / fp_t and Montgomery constants now live in
// cuda-common. This file is a shim: it pulls the canonical device path
// from cuda-common and retains the host-side blst-based field types in
// place so NVCC's host-compile pass over kernel headers — which see
// `typedef fr_t scalar_t;` outside __CUDA_ARCH__ — still finds concrete
// types. Replacing the host blst path is a follow-up.

#include "ff/alt_bn128.cuh"

#ifndef __CUDA_ARCH__   // host-side field types
#include "../blst/blst_t.hpp"

static const vec256 ALT_BN128_P = {
    TO_LIMB_T(0x3c208c16d87cfd47), TO_LIMB_T(0x97816a916871ca8d),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_RR = {    /* (1<<512)%P */
    TO_LIMB_T(0xf32cfc5b538afa89), TO_LIMB_T(0xb5e71911d44501fb),
    TO_LIMB_T(0x47ab1eff0a417ff6), TO_LIMB_T(0x06d89f71cab8351f),
};
static const vec256 ALT_BN128_ONE = {   /* (1<<256)%P */
    TO_LIMB_T(0xd35d438dc58f0d9d), TO_LIMB_T(0x0a78eb28f5c70b3d),
    TO_LIMB_T(0x666ea36f7879462c), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<ALT_BN128_P, 0x87d20782e4866389u,
                   ALT_BN128_RR, ALT_BN128_ONE> fp_t;

static const vec256 ALT_BN128_r = {
    TO_LIMB_T(0x43e1f593f0000001), TO_LIMB_T(0x2833e84879b97091),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_rRR = {   /* (1<<512)%r */
    TO_LIMB_T(0x1bb8e645ae216da7), TO_LIMB_T(0x53fe3ab1e35c59e3),
    TO_LIMB_T(0x8c49833d53bb8085), TO_LIMB_T(0x0216d0b17f4e44a5)
};
static const vec256 ALT_BN128_rONE = {  /* (1<<256)%r */
    TO_LIMB_T(0xac96341c4ffffffb), TO_LIMB_T(0x36fc76959f60cd29),
    TO_LIMB_T(0x666ea36f7879462e), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<ALT_BN128_r, 0xc2e1f593efffffffu,
                   ALT_BN128_rRR, ALT_BN128_rONE> fr_t;
#endif
