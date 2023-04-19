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
#include "cpu_math.h"
#include "structures.h"


#define G_X _uint256{0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798}
#define G_Y _uint256{0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8}
#define G CurvePoint{G_X, G_Y}


_uint256 cpu_point_double_lambda(CurvePoint p) {
    return cpu_mul_256_mod_p(cpu_mul_256_mod_p(_uint256{0, 0, 0, 0, 0, 0, 0, 3}, cpu_mul_256_mod_p(p.x, p.x)), cpu_eeuclid_256_mod_p(cpu_mul_256_mod_p(_uint256{0, 0, 0, 0, 0, 0, 0, 2}, p.y)));
}

_uint256 cpu_point_add_lambda(CurvePoint p, CurvePoint q) {
    return cpu_mul_256_mod_p(cpu_sub_256_mod_p(q.y, p.y), cpu_eeuclid_256_mod_p(cpu_sub_256_mod_p(q.x, p.x)));
}

CurvePoint cpu_point_add(CurvePoint p, CurvePoint q) {
    _uint256 lambda;
    if (eqeq_256(p.x, q.x)) {
        lambda = cpu_point_double_lambda(p);
    } else {
        lambda = cpu_point_add_lambda(p, q);
    }

    CurvePoint r;
    r.x = cpu_sub_256_mod_p(cpu_sub_256_mod_p(cpu_mul_256_mod_p(lambda, lambda), p.x), q.x);
    r.y = cpu_sub_256_mod_p(cpu_mul_256_mod_p(lambda, cpu_sub_256_mod_p(p.x, r.x)), p.y);
    return r;
}


CurvePoint cpu_point_multiply(CurvePoint x, _uint256 y) {
    CurvePoint result;
    bool at_infinity = true;
    CurvePoint temp = x;

    for (int i = 0; i < 32; i++) {
        if ((y.h & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.g & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.f & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.e & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.d & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.c & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.b & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    for (int i = 0; i < 32; i++) {
        if ((y.a & (1ULL << i))) {
            at_infinity ? (result = temp) : (result = cpu_point_add(result, temp));
            at_infinity = false;
        }
        temp = cpu_point_add(temp, temp);
    }

    return result;
}