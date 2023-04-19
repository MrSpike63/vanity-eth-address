/*
    Copyright (C) 2023 MrSpike63

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, version 3.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#pragma once
#include <cinttypes>

#include "structures.h"


#define P _uint256{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F}


__device__ _uint256c add_256_with_c(_uint256 x, _uint256 y) {
    _uint256c result;

    uint32_t carry = 0;
    asm(
        "add.cc.u32 %0, %9, %17;    \n\t"
        "addc.cc.u32 %1, %10, %18;  \n\t"
        "addc.cc.u32 %2, %11, %19;  \n\t"
        "addc.cc.u32 %3, %12, %20;  \n\t"
        "addc.cc.u32 %4, %13, %21;  \n\t"
        "addc.cc.u32 %5, %14, %22;  \n\t"
        "addc.cc.u32 %6, %15, %23;  \n\t"
        "addc.cc.u32 %7, %16, %24;  \n\t"
        "addc.u32 %8, 0x0, 0x0;     \n\t"
        : "=r"(result.h), "=r"(result.g), "=r"(result.f), "=r"(result.e), "=r"(result.d), "=r"(result.c), "=r"(result.b), "=r"(result.a), "=r"(carry) : "r"(x.h), "r"(x.g), "r"(x.f), "r"(x.e), "r"(x.d), "r"(x.c), "r"(x.b), "r"(x.a), "r"(y.h), "r"(y.g), "r"(y.f), "r"(y.e), "r"(y.d), "r"(y.c), "r"(y.b), "r"(y.a)
    );
   
    result.carry = (bool)carry;

    return result;
}


__device__ _uint256 sub_256(_uint256 x, _uint256 y) {
    _uint256 result;

    asm(
        "sub.cc.u32 %0, %8, %16;    \n\t"
        "subc.cc.u32 %1, %9, %17;   \n\t"
        "subc.cc.u32 %2, %10, %18;  \n\t"
        "subc.cc.u32 %3, %11, %19;  \n\t"
        "subc.cc.u32 %4, %12, %20;  \n\t"
        "subc.cc.u32 %5, %13, %21;  \n\t"
        "subc.cc.u32 %6, %14, %22;  \n\t"
        "subc.u32 %7, %15, %23;     \n\t"
        : "=r"(result.h), "=r"(result.g), "=r"(result.f), "=r"(result.e), "=r"(result.d), "=r"(result.c), "=r"(result.b), "=r"(result.a) : "r"(x.h), "r"(x.g), "r"(x.f), "r"(x.e), "r"(x.d), "r"(x.c), "r"(x.b), "r"(x.a), "r"(y.h), "r"(y.g), "r"(y.f), "r"(y.e), "r"(y.d), "r"(y.c), "r"(y.b), "r"(y.a)
    );

    return result;
}


__device__ _uint256 sub_256_mod_p(_uint256 x, _uint256 y) {
    _uint256 result;

    asm(
        "{\n\t"
        ".reg .u32 t;                       \n\t"
        ".reg .pred p;                      \n\t"
        "sub.cc.u32 %0, %8, %16;            \n\t"
        "subc.cc.u32 %1, %9, %17;           \n\t"
        "subc.cc.u32 %2, %10, %18;          \n\t"
        "subc.cc.u32 %3, %11, %19;          \n\t"
        "subc.cc.u32 %4, %12, %20;          \n\t"
        "subc.cc.u32 %5, %13, %21;          \n\t"
        "subc.cc.u32 %6, %14, %22;          \n\t"
        "subc.cc.u32 %7, %15, %23;          \n\t"
        "subc.u32 t, 0x0, 0x0;              \n\t"
        "setp.ne.u32 p, t, 0x0;             \n\t"
        "@p add.cc.u32 %0, %0, 0xFFFFFC2F;  \n\t"
        "@p addc.cc.u32 %1, %1, 0xFFFFFFFE; \n\t"
        "@p addc.cc.u32 %2, %2, 0xFFFFFFFF; \n\t"
        "@p addc.cc.u32 %3, %3, 0xFFFFFFFF; \n\t"
        "@p addc.cc.u32 %4, %4, 0xFFFFFFFF; \n\t"
        "@p addc.cc.u32 %5, %5, 0xFFFFFFFF; \n\t"
        "@p addc.cc.u32 %6, %6, 0xFFFFFFFF; \n\t"
        "@p addc.u32 %7, %7, 0xFFFFFFFF;    \n\t"
        "}"
        : "=r"(result.h), "=r"(result.g), "=r"(result.f), "=r"(result.e), "=r"(result.d), "=r"(result.c), "=r"(result.b), "=r"(result.a) : "r"(x.h), "r"(x.g), "r"(x.f), "r"(x.e), "r"(x.d), "r"(x.c), "r"(x.b), "r"(x.a), "r"(y.h), "r"(y.g), "r"(y.f), "r"(y.e), "r"(y.d), "r"(y.c), "r"(y.b), "r"(y.a)
    );

    return result;
}


__device__ _uint256 mul_256_mod_p(_uint256 x, _uint256 y) {
    _uint256 r{0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t carry = 0;

    asm(
        "{\n\t"
        ".reg .u32 a;                       \n\t"
        ".reg .u32 b;                       \n\t"
        ".reg .u32 c;                       \n\t"
        ".reg .u32 d;                       \n\t"
        ".reg .u32 e;                       \n\t"
        ".reg .u32 f;                       \n\t"
        ".reg .u32 g;                       \n\t"
        ".reg .u32 h;                       \n\t"
        ".reg .u32 i;                       \n\t"
        ".reg .u32 j;                       \n\t"
        ".reg .u32 k;                       \n\t"
        ".reg .u32 l;                       \n\t"
        ".reg .u32 m;                       \n\t"
        ".reg .u32 n;                       \n\t"
        ".reg .u32 o;                       \n\t"
        ".reg .u32 p;                       \n\t"
        ".reg .u32 t1;                      \n\t"
        "mul.lo.u32 p, %9, %17;             \n\t"
        "mul.hi.u32 t1, %9, %17;            \n\t"
        "mad.lo.cc.u32 o, %10, %17, t1;     \n\t"
        "mul.hi.u32 t1, %10, %17;           \n\t"
        "madc.lo.cc.u32 n, %11, %17, t1;    \n\t"
        "mul.hi.u32 t1, %11, %17;           \n\t"
        "madc.lo.cc.u32 m, %12, %17, t1;    \n\t"
        "mul.hi.u32 t1, %12, %17;           \n\t"
        "madc.lo.cc.u32 l, %13, %17, t1;    \n\t"
        "mul.hi.u32 t1, %13, %17;           \n\t"
        "madc.lo.cc.u32 k, %14, %17, t1;    \n\t"
        "mul.hi.u32 t1, %14, %17;           \n\t"
        "madc.lo.cc.u32 j, %15, %17, t1;    \n\t"
        "mul.hi.u32 t1, %15, %17;           \n\t"
        "madc.lo.cc.u32 i, %16, %17, t1;    \n\t"
        "madc.hi.u32 h, %16, %17, 0x0;      \n\t"
        "mad.lo.cc.u32 o, %9, %18, o;       \n\t"
        "madc.lo.cc.u32 n, %10, %18, n;     \n\t"
        "madc.lo.cc.u32 m, %11, %18, m;     \n\t"
        "madc.lo.cc.u32 l, %12, %18, l;     \n\t"
        "madc.lo.cc.u32 k, %13, %18, k;     \n\t"
        "madc.lo.cc.u32 j, %14, %18, j;     \n\t"
        "madc.lo.cc.u32 i, %15, %18, i;     \n\t"
        "madc.lo.cc.u32 h, %16, %18, h;     \n\t"
        "addc.u32 g, 0x0, 0x0;              \n\t"
        "mad.hi.cc.u32 n, %9, %18, n;       \n\t"
        "madc.hi.cc.u32 m, %10, %18, m;     \n\t"
        "madc.hi.cc.u32 l, %11, %18, l;     \n\t"
        "madc.hi.cc.u32 k, %12, %18, k;     \n\t"
        "madc.hi.cc.u32 j, %13, %18, j;     \n\t"
        "madc.hi.cc.u32 i, %14, %18, i;     \n\t"
        "madc.hi.cc.u32 h, %15, %18, h;     \n\t"
        "madc.hi.cc.u32 g, %16, %18, g;     \n\t"
        "addc.u32 f, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 n, %9, %19, n;       \n\t"
        "madc.lo.cc.u32 m, %10, %19, m;     \n\t"
        "madc.lo.cc.u32 l, %11, %19, l;     \n\t"
        "madc.lo.cc.u32 k, %12, %19, k;     \n\t"
        "madc.lo.cc.u32 j, %13, %19, j;     \n\t"
        "madc.lo.cc.u32 i, %14, %19, i;     \n\t"
        "madc.lo.cc.u32 h, %15, %19, h;     \n\t"
        "madc.lo.cc.u32 g, %16, %19, g;     \n\t"
        "addc.u32 f, f, 0x0;                \n\t"
        "mad.hi.cc.u32 m, %9, %19, m;       \n\t"
        "madc.hi.cc.u32 l, %10, %19, l;     \n\t"
        "madc.hi.cc.u32 k, %11, %19, k;     \n\t"
        "madc.hi.cc.u32 j, %12, %19, j;     \n\t"
        "madc.hi.cc.u32 i, %13, %19, i;     \n\t"
        "madc.hi.cc.u32 h, %14, %19, h;     \n\t"
        "madc.hi.cc.u32 g, %15, %19, g;     \n\t"
        "madc.hi.cc.u32 f, %16, %19, f;     \n\t"
        "addc.u32 e, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 m, %9, %20, m;       \n\t"
        "madc.lo.cc.u32 l, %10, %20, l;     \n\t"
        "madc.lo.cc.u32 k, %11, %20, k;     \n\t"
        "madc.lo.cc.u32 j, %12, %20, j;     \n\t"
        "madc.lo.cc.u32 i, %13, %20, i;     \n\t"
        "madc.lo.cc.u32 h, %14, %20, h;     \n\t"
        "madc.lo.cc.u32 g, %15, %20, g;     \n\t"
        "madc.lo.cc.u32 f, %16, %20, f;     \n\t"
        "addc.u32 e, e, 0x0;                \n\t"
        "mad.hi.cc.u32 l, %9, %20, l;       \n\t"
        "madc.hi.cc.u32 k, %10, %20, k;     \n\t"
        "madc.hi.cc.u32 j, %11, %20, j;     \n\t"
        "madc.hi.cc.u32 i, %12, %20, i;     \n\t"
        "madc.hi.cc.u32 h, %13, %20, h;     \n\t"
        "madc.hi.cc.u32 g, %14, %20, g;     \n\t"
        "madc.hi.cc.u32 f, %15, %20, f;     \n\t"
        "madc.hi.cc.u32 e, %16, %20, e;     \n\t"
        "addc.u32 d, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 l, %9, %21, l;       \n\t"
        "madc.lo.cc.u32 k, %10, %21, k;     \n\t"
        "madc.lo.cc.u32 j, %11, %21, j;     \n\t"
        "madc.lo.cc.u32 i, %12, %21, i;     \n\t"
        "madc.lo.cc.u32 h, %13, %21, h;     \n\t"
        "madc.lo.cc.u32 g, %14, %21, g;     \n\t"
        "madc.lo.cc.u32 f, %15, %21, f;     \n\t"
        "madc.lo.cc.u32 e, %16, %21, e;     \n\t"
        "addc.u32 d, d, 0x0;                \n\t"
        "mad.hi.cc.u32 k, %9, %21, k;       \n\t"
        "madc.hi.cc.u32 j, %10, %21, j;     \n\t"
        "madc.hi.cc.u32 i, %11, %21, i;     \n\t"
        "madc.hi.cc.u32 h, %12, %21, h;     \n\t"
        "madc.hi.cc.u32 g, %13, %21, g;     \n\t"
        "madc.hi.cc.u32 f, %14, %21, f;     \n\t"
        "madc.hi.cc.u32 e, %15, %21, e;     \n\t"
        "madc.hi.cc.u32 d, %16, %21, d;     \n\t"
        "addc.u32 c, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 k, %9, %22, k;       \n\t"
        "madc.lo.cc.u32 j, %10, %22, j;     \n\t"
        "madc.lo.cc.u32 i, %11, %22, i;     \n\t"
        "madc.lo.cc.u32 h, %12, %22, h;     \n\t"
        "madc.lo.cc.u32 g, %13, %22, g;     \n\t"
        "madc.lo.cc.u32 f, %14, %22, f;     \n\t"
        "madc.lo.cc.u32 e, %15, %22, e;     \n\t"
        "madc.lo.cc.u32 d, %16, %22, d;     \n\t"
        "addc.u32 c, c, 0x0;                \n\t"
        "mad.hi.cc.u32 j, %9, %22, j;       \n\t"
        "madc.hi.cc.u32 i, %10, %22, i;     \n\t"
        "madc.hi.cc.u32 h, %11, %22, h;     \n\t"
        "madc.hi.cc.u32 g, %12, %22, g;     \n\t"
        "madc.hi.cc.u32 f, %13, %22, f;     \n\t"
        "madc.hi.cc.u32 e, %14, %22, e;     \n\t"
        "madc.hi.cc.u32 d, %15, %22, d;     \n\t"
        "madc.hi.cc.u32 c, %16, %22, c;     \n\t"
        "addc.u32 b, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 j, %9, %23, j;       \n\t"
        "madc.lo.cc.u32 i, %10, %23, i;     \n\t"
        "madc.lo.cc.u32 h, %11, %23, h;     \n\t"
        "madc.lo.cc.u32 g, %12, %23, g;     \n\t"
        "madc.lo.cc.u32 f, %13, %23, f;     \n\t"
        "madc.lo.cc.u32 e, %14, %23, e;     \n\t"
        "madc.lo.cc.u32 d, %15, %23, d;     \n\t"
        "madc.lo.cc.u32 c, %16, %23, c;     \n\t"
        "addc.u32 b, b, 0x0;                \n\t"
        "mad.hi.cc.u32 i, %9, %23, i;       \n\t"
        "madc.hi.cc.u32 h, %10, %23, h;     \n\t"
        "madc.hi.cc.u32 g, %11, %23, g;     \n\t"
        "madc.hi.cc.u32 f, %12, %23, f;     \n\t"
        "madc.hi.cc.u32 e, %13, %23, e;     \n\t"
        "madc.hi.cc.u32 d, %14, %23, d;     \n\t"
        "madc.hi.cc.u32 c, %15, %23, c;     \n\t"
        "madc.hi.cc.u32 b, %16, %23, b;     \n\t"
        "addc.u32 a, 0x0, 0x0;              \n\t"
        "mad.lo.cc.u32 i, %9, %24, i;       \n\t"
        "madc.lo.cc.u32 h, %10, %24, h;     \n\t"
        "madc.lo.cc.u32 g, %11, %24, g;     \n\t"
        "madc.lo.cc.u32 f, %12, %24, f;     \n\t"
        "madc.lo.cc.u32 e, %13, %24, e;     \n\t"
        "madc.lo.cc.u32 d, %14, %24, d;     \n\t"
        "madc.lo.cc.u32 c, %15, %24, c;     \n\t"
        "madc.lo.cc.u32 b, %16, %24, b;     \n\t"
        "addc.u32 a, a, 0x0;                \n\t"
        "mad.hi.cc.u32 h, %9, %24, h;       \n\t"
        "madc.hi.cc.u32 g, %10, %24, g;     \n\t"
        "madc.hi.cc.u32 f, %11, %24, f;     \n\t"
        "madc.hi.cc.u32 e, %12, %24, e;     \n\t"
        "madc.hi.cc.u32 d, %13, %24, d;     \n\t"
        "madc.hi.cc.u32 c, %14, %24, c;     \n\t"
        "madc.hi.cc.u32 b, %15, %24, b;     \n\t"
        "madc.hi.u32 a, %16, %24, a;        \n\t"
        ".reg.u32 ov;                       \n\t"
        "mul.lo.u32 %0, h, 0x3d1;           \n\t"
        "mul.hi.u32 t1, h, 0x3d1;           \n\t"
        "mad.lo.cc.u32 %1, g, 0x3d1, t1;    \n\t"
        "mul.hi.u32 t1, g, 0x3d1;           \n\t"
        "madc.lo.cc.u32 %2, f, 0x3d1, t1;   \n\t"
        "mul.hi.u32 t1, f, 0x3d1;           \n\t"
        "madc.lo.cc.u32 %3, e, 0x3d1, t1;   \n\t"
        "mul.hi.u32 t1, e, 0x3d1;           \n\t"
        "madc.lo.cc.u32 %4, d, 0x3d1, t1;   \n\t"
        "mul.hi.u32 t1, d, 0x3d1;           \n\t"
        "madc.lo.cc.u32 %5, c, 0x3d1, t1;   \n\t"
        "mul.hi.u32 t1, c, 0x3d1;           \n\t"
        "madc.lo.cc.u32 %6, b, 0x3d1, t1;   \n\t"
        "madc.hi.u32 %7, b, 0x3d1, 0x0;     \n\t"
        "add.cc.u32 %1, %1, h;              \n\t"
        "addc.cc.u32 %2, %2, g;             \n\t"
        "addc.cc.u32 %3, %3, f;             \n\t"
        "addc.cc.u32 %4, %4, e;             \n\t"
        "addc.cc.u32 %5, %5, d;             \n\t"
        "addc.cc.u32 %6, %6, c;             \n\t"
        "addc.cc.u32 %7, %7, b;             \n\t"
        "addc.u32 ov, 0x0, 0x0;             \n\t"
        ".reg .u32 n1;                      \n\t"
        ".reg .u32 n2;                      \n\t"
        ".reg .u32 n3;                      \n\t"
        "mul.lo.u32 n3, a, 0x3d1;           \n\t"
        "mad.hi.cc.u32 n2, a, 0x3d1, a;     \n\t"
        "addc.u32 n1, 0x0, 0x0;             \n\t"
        "add.cc.u32 %0, %0, n3;             \n\t"
        "addc.cc.u32 %1, %1, n2;            \n\t"
        "addc.cc.u32 %2, %2, n1;            \n\t"
        "addc.cc.u32 %3, %3, 0x0;           \n\t"
        "addc.cc.u32 %4, %4, 0x0;           \n\t"
        "addc.cc.u32 %5, %5, 0x0;           \n\t"
        "addc.cc.u32 %6, %6, 0x0;           \n\t"
        "madc.lo.cc.u32 %7, a, 0x3d1, %7;   \n\t"
        "madc.hi.u32 ov, a, 0x3d1, ov;      \n\t"
        "add.cc.u32 %0, %0, p;              \n\t"
        "addc.cc.u32 %1, %1, o;             \n\t"
        "addc.cc.u32 %2, %2, n;             \n\t"
        "addc.cc.u32 %3, %3, m;             \n\t"
        "addc.cc.u32 %4, %4, l;             \n\t"
        "addc.cc.u32 %5, %5, k;             \n\t"
        "addc.cc.u32 %6, %6, j;             \n\t"
        "addc.cc.u32 %7, %7, i;             \n\t"
        "addc.u32 ov, ov, 0x0;              \n\t"
        "mul.lo.u32 n2, ov, 0x3d1;          \n\t"
        "mad.hi.u32 n1, ov, 0x3d1, ov;      \n\t"
        "add.cc.u32 %0, %0, n2;             \n\t"
        "addc.cc.u32 %1, %1, n1;            \n\t"
        "addc.cc.u32 %2, %2, 0x0;           \n\t"
        "addc.cc.u32 %3, %3, 0x0;           \n\t"
        "addc.cc.u32 %4, %4, 0x0;           \n\t"
        "addc.cc.u32 %5, %5, 0x0;           \n\t"
        "addc.cc.u32 %6, %6, 0x0;           \n\t"
        "addc.cc.u32 %7, %7, 0x0;           \n\t"
        "addc.u32 %8, 0x0, 0x0;             \n\t"
        "}"
        : "+r"(r.h), "+r"(r.g), "+r"(r.f), "+r"(r.e), "+r"(r.d), "+r"(r.c), "+r"(r.b), "+r"(r.a), "=r"(carry) : "r"(x.h), "r"(x.g), "r"(x.f), "r"(x.e), "r"(x.d), "r"(x.c), "r"(x.b), "r"(x.a), "r"(y.h), "r"(y.g), "r"(y.f), "r"(y.e), "r"(y.d), "r"(y.c), "r"(y.b), "r"(y.a)
    );

    bool overflow = carry || gte_256(r, P);
    if (overflow) {
        r = sub_256(r, P);
    }

    return r;
}


__device__ _uint256 rshift1_256(_uint256 x) {
    _uint256 result;
    result.a =               (x.a >> 1);
    result.b = (x.a << 31) | (x.b >> 1);
    result.c = (x.b << 31) | (x.c >> 1);
    result.d = (x.c << 31) | (x.d >> 1);
    result.e = (x.d << 31) | (x.e >> 1);
    result.f = (x.e << 31) | (x.f >> 1);
    result.g = (x.f << 31) | (x.g >> 1);
    result.h = (x.g << 31) | (x.h >> 1);
    return result;
}

__device__ _uint256 rshift1_256c(_uint256c x) {
    _uint256 result;
    result.a = ((uint32_t)x.carry << 31) | (x.a >> 1);
    result.b = (x.a << 31) | (x.b >> 1);
    result.c = (x.b << 31) | (x.c >> 1);
    result.d = (x.c << 31) | (x.d >> 1);
    result.e = (x.d << 31) | (x.e >> 1);
    result.f = (x.e << 31) | (x.f >> 1);
    result.g = (x.f << 31) | (x.g >> 1);
    result.h = (x.g << 31) | (x.h >> 1);
    return result;
}


__device__ _uint256 eeuclid_256_mod_p(_uint256 input) {
    _uint256 u = input;
    _uint256 v = P;
    _uint256 x{0, 0, 0, 0, 0, 0, 0, 1};
    _uint256 y{0, 0, 0, 0, 0, 0, 0, 0};

    while ((u.h & 1) == 0) {
        u = rshift1_256(u);

        _uint256c x_;
        if ((x.h & 1) == 1) {
            x_ = add_256_with_c(x, P);
        } else {
            x_ = uint256_to_uint256c(x);
        }
        x = rshift1_256c(x_);
    }

    bool prmt = false;
    while (true) {
        bool gt = false;
        bool equal = true;
        gt |= (u.a > v.a); equal &= (u.a == v.a);
        gt |= ((u.b > v.b) && equal); equal &= (u.b == v.b);
        gt |= ((u.c > v.c) && equal); equal &= (u.c == v.c);
        gt |= ((u.d > v.d) && equal); equal &= (u.d == v.d);
        gt |= ((u.e > v.e) && equal); equal &= (u.e == v.e);
        gt |= ((u.f > v.f) && equal); equal &= (u.f == v.f);
        gt |= ((u.g > v.g) && equal); equal &= (u.g == v.g);
        gt |= ((u.h > v.h) && equal); equal &= (u.h == v.h);

        if (equal) { break; }
        if (gt) {
            prmt = !prmt;
            _uint256 t = u;
            u = v;
            v = t;
            t = x;
            x = y;
            y = t;
        }

        v = sub_256(v, u);
        y = sub_256_mod_p(y, x);

        while ((v.h & 1) == 0) {
            v = rshift1_256(v);

            _uint256c y_;
            if ((y.h & 1) == 1) {
                y_ = add_256_with_c(y, P);
            } else {
                y_ = uint256_to_uint256c(y);
            }
            y = rshift1_256c(y_);
        }
    }

    return prmt ? y : x;
}