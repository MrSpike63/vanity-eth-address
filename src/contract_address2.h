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


__global__ void __launch_bounds__(BLOCK_SIZE, 2) gpu_contract2_address_work(int score_method, Address a, _uint256 base_key, _uint256 bytecode) {
    uint32_t thread_id = (uint32_t)threadIdx.x + (uint32_t)blockIdx.x * BLOCK_SIZE;

    _uint256 key = base_key;
    key.h += THREAD_WORK * thread_id;
    for (int i = 0; i < THREAD_WORK; i++) {
        handle_output2(score_method, calculate_contract_address2(a, key, bytecode), key.h - base_key.h);
        key.h += 1;
    }
}