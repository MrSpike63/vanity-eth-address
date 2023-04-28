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
#include "curve_math.h"
#include "keccak.h"
#include "math.h"


__global__ void __launch_bounds__(BLOCK_SIZE, 2) gpu_contract_address_work(int score_method, CurvePoint* offsets) {
    bool b = __isGlobal(offsets);
    __builtin_assume(b);

    uint64_t thread_id = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)BLOCK_SIZE;
    uint64_t key = (uint64_t)THREAD_WORK * thread_id;

    CurvePoint p = offsets[thread_id];

    handle_output(score_method, calculate_contract_address(calculate_address(p.x, p.y)), key, 0);
    handle_output(score_method, calculate_contract_address(calculate_address(p.x, sub_256(P, p.y))), key, 1);


    _uint256 z[THREAD_WORK - 1];
    z[0] = sub_256_mod_p(p.x, addends[0].x);

    for (int i = 1; i < THREAD_WORK - 1; i++) {
        _uint256 x_delta = sub_256_mod_p(p.x, addends[i].x);
        z[i] = mul_256_mod_p(z[i - 1], x_delta);
    }

    _uint256 q = eeuclid_256_mod_p(z[THREAD_WORK - 2]);

    for (int i = THREAD_WORK - 2; i >= 1; i--) {
        _uint256 y = mul_256_mod_p(q, z[i - 1]);
        q = mul_256_mod_p(q, sub_256_mod_p(p.x, addends[i].x));

        _uint256 lambda = mul_256_mod_p(sub_256_mod_p(p.y, addends[i].y), y);
        _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), p.x), addends[i].x);
        _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(p.x, curve_x)), p.y);

        handle_output(score_method, calculate_contract_address(calculate_address(curve_x, curve_y)), key + i + 1, 0);
        handle_output(score_method, calculate_contract_address(calculate_address(curve_x, sub_256(P, curve_y))), key + i + 1, 1);
    }

    _uint256 y = q;

    _uint256 lambda = mul_256_mod_p(sub_256_mod_p(p.y, addends[0].y), y);
    _uint256 curve_x = sub_256_mod_p(sub_256_mod_p(mul_256_mod_p(lambda, lambda), p.x), addends[0].x);
    _uint256 curve_y = sub_256_mod_p(mul_256_mod_p(lambda, sub_256_mod_p(p.x, curve_x)), p.y);

    handle_output(score_method, calculate_contract_address(calculate_address(curve_x, curve_y)), key + 1, 0);
    handle_output(score_method, calculate_contract_address(calculate_address(curve_x, sub_256(P, curve_y))), key + 1, 1);
}