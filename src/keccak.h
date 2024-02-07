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


__device__ uint64_t rotate(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

__device__ uint64_t swap_endianness(uint64_t x) {
    return ((x & 0x00000000000000FF) << 56) | ((x & 0x000000000000FF00) << 40) | ((x & 0x0000000000FF0000) << 24) | ((x & 0x00000000FF000000) << 8) | ((x & 0x000000FF00000000) >> 8) | ((x & 0x0000FF0000000000) >> 24) | ((x & 0x00FF000000000000) >> 40) | ((x & 0xFF00000000000000) >> 56);
}

__constant__ uint64_t IOTA_CONSTANTS[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};


__device__ void block_permute(uint64_t *block) {
    uint64_t C[5];
    uint64_t temp1, temp2;

    for (int t = 0; t < 24; t++) {
        C[0] = block[0] ^ block[1] ^ block[2] ^ block[3] ^ block[4];
        C[1] = block[5] ^ block[6] ^ block[7] ^ block[8] ^ block[9];
        C[2] = block[10] ^ block[11] ^ block[12] ^ block[13] ^ block[14];
        C[3] = block[15] ^ block[16] ^ block[17] ^ block[18] ^ block[19];
        C[4] = block[20] ^ block[21] ^ block[22] ^ block[23] ^ block[24];

        block[0] ^= C[4] ^ rotate(C[1], 1); block[1] ^= C[4] ^ rotate(C[1], 1); block[2] ^= C[4] ^ rotate(C[1], 1); block[3] ^= C[4] ^ rotate(C[1], 1); block[4] ^= C[4] ^ rotate(C[1], 1);
        block[5] ^= C[0] ^ rotate(C[2], 1); block[6] ^= C[0] ^ rotate(C[2], 1); block[7] ^= C[0] ^ rotate(C[2], 1); block[8] ^= C[0] ^ rotate(C[2], 1); block[9] ^= C[0] ^ rotate(C[2], 1);
        block[10] ^= C[1] ^ rotate(C[3], 1); block[11] ^= C[1] ^ rotate(C[3], 1); block[12] ^= C[1] ^ rotate(C[3], 1); block[13] ^= C[1] ^ rotate(C[3], 1); block[14] ^= C[1] ^ rotate(C[3], 1);
        block[15] ^= C[2] ^ rotate(C[4], 1); block[16] ^= C[2] ^ rotate(C[4], 1); block[17] ^= C[2] ^ rotate(C[4], 1); block[18] ^= C[2] ^ rotate(C[4], 1); block[19] ^= C[2] ^ rotate(C[4], 1);
        block[20] ^= C[3] ^ rotate(C[0], 1); block[21] ^= C[3] ^ rotate(C[0], 1); block[22] ^= C[3] ^ rotate(C[0], 1); block[23] ^= C[3] ^ rotate(C[0], 1); block[24] ^= C[3] ^ rotate(C[0], 1);


        temp1 = block[8];
        block[8] = rotate(block[1], 36);  
        block[1] = rotate(block[15], 28); 
        block[15] = rotate(block[18], 21);
        block[18] = rotate(block[13], 15);
        block[13] = rotate(block[7], 10); 
        block[7] = rotate(block[11], 6);  
        block[11] = rotate(block[2], 3);  
        block[2] = rotate(block[5], 1);   
        block[5] = rotate(block[6], 44);  
        block[6] = rotate(block[21], 20); 
        block[21] = rotate(block[14], 61);
        block[14] = rotate(block[22], 39);
        block[22] = rotate(block[4], 18);
        block[4] = rotate(block[10], 62);
        block[10] = rotate(block[12], 43);
        block[12] = rotate(block[17], 25);
        block[17] = rotate(block[23], 8);
        block[23] = rotate(block[19], 56);
        block[19] = rotate(block[3], 41);
        block[3] = rotate(block[20], 27);
        block[20] = rotate(block[24], 14);
        block[24] = rotate(block[9], 2);
        block[9] = rotate(block[16], 55);
        block[16] = rotate(temp1, 45);


        temp1 = block[0];
        temp2 = block[5];
        block[0] ^= (~block[5] & block[10]);
        block[5] ^= (~block[10] & block[15]);
        block[10] ^= (~block[15] & block[20]);
        block[15] ^= (~block[20] & temp1);
        block[20] ^= (~temp1 & temp2);

        temp1 = block[1];
        temp2 = block[6];
        block[1] ^= (~block[6] & block[11]);
        block[6] ^= (~block[11] & block[16]);
        block[11] ^= (~block[16] & block[21]);
        block[16] ^= (~block[21] & temp1);
        block[21] ^= (~temp1 & temp2);

        temp1 = block[2];
        temp2 = block[7];
        block[2] ^= (~block[7] & block[12]);
        block[7] ^= (~block[12] & block[17]);
        block[12] ^= (~block[17] & block[22]);
        block[17] ^= (~block[22] & temp1);
        block[22] ^= (~temp1 & temp2);

        temp1 = block[3];
        temp2 = block[8];
        block[3] ^= (~block[8] & block[13]);
        block[8] ^= (~block[13] & block[18]);
        block[13] ^= (~block[18] & block[23]);
        block[18] ^= (~block[23] & temp1);
        block[23] ^= (~temp1 & temp2);

        temp1 = block[4];
        temp2 = block[9];
        block[4] ^= (~block[9] & block[14]);
        block[9] ^= (~block[14] & block[19]);
        block[14] ^= (~block[19] & block[24]);
        block[19] ^= (~block[24] & temp1);
        block[24] ^= (~temp1 & temp2);


        block[0] ^= IOTA_CONSTANTS[t];
    }
}


__device__ Address calculate_address(_uint256 x, _uint256 y) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = swap_endianness(((uint64_t)x.a << 32) | x.b);
    block[5] = swap_endianness(((uint64_t)x.c << 32) | x.d);
    block[10] = swap_endianness(((uint64_t)x.e << 32) | x.f);
    block[15] = swap_endianness(((uint64_t)x.g << 32) | x.h);
    block[20] = swap_endianness(((uint64_t)y.a << 32) | y.b);
    block[1] = swap_endianness(((uint64_t)y.c << 32) | y.d);
    block[6] = swap_endianness(((uint64_t)y.e << 32) | y.f);
    block[11] = swap_endianness(((uint64_t)y.g << 32) | y.h);
    block[16] = (1ULL << 0);

    block[8] = 0x8000000000000000;

    block_permute(block);

    uint64_t b = swap_endianness(block[5]);
    uint64_t c = swap_endianness(block[10]);
    uint64_t d = swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}


__device__ Address calculate_contract_address(Address a, uint8_t nonce = 0x80) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = swap_endianness((0xD694ULL << 48) | ((uint64_t)a.a << 16) | (a.b >> 16));
    block[5] = swap_endianness(((uint64_t)a.b << 48) | ((uint64_t)a.c << 16) | (a.d >> 16));
    block[10] = swap_endianness(((uint64_t)a.d << 48) | ((uint64_t)a.e << 16) | ((uint64_t)nonce << 8) | 1);

    block[8] = 0x8000000000000000;

    block_permute(block);

    uint64_t b = swap_endianness(block[5]);
    uint64_t c = swap_endianness(block[10]);
    uint64_t d = swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}


__device__ Address calculate_contract_address2(Address a, _uint256 salt, _uint256 bytecode) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = swap_endianness((0xFFULL << 56) | ((uint64_t)a.a << 24) | (a.b >> 8));
    block[5] = swap_endianness(((uint64_t)a.b << 56) | ((uint64_t)a.c << 24) | (a.d >> 8));
    block[10] = swap_endianness(((uint64_t)a.d << 56) | ((uint64_t)a.e << 24) | (salt.a >> 8));
    block[15] = swap_endianness(((uint64_t)salt.a << 56) | ((uint64_t)salt.b << 24) | (salt.c >> 8));
    block[20] = swap_endianness(((uint64_t)salt.c << 56) | ((uint64_t)salt.d << 24) | (salt.e >> 8));
    block[1] = swap_endianness(((uint64_t)salt.e << 56) | ((uint64_t)salt.f << 24) | (salt.g >> 8));
    block[6] = swap_endianness(((uint64_t)salt.g << 56) | ((uint64_t)salt.h << 24) | (bytecode.a >> 8));
    block[11] = swap_endianness(((uint64_t)bytecode.a << 56) | ((uint64_t)bytecode.b << 24) | (bytecode.c >> 8));
    block[16] = swap_endianness(((uint64_t)bytecode.c << 56) | ((uint64_t)bytecode.d << 24) | (bytecode.e >> 8));
    block[21] = swap_endianness(((uint64_t)bytecode.e << 56) | ((uint64_t)bytecode.f << 24) | (bytecode.g >> 8));
    block[2] = swap_endianness(((uint64_t)bytecode.g << 56) | ((uint64_t)bytecode.h << 24) | (1 << 16));

    block[8] = 0x8000000000000000;

    block_permute(block);

    uint64_t b = swap_endianness(block[5]);
    uint64_t c = swap_endianness(block[10]);
    uint64_t d = swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}

__device__ _uint256 calculate_create3_salt(Address origin, _uint256 salt) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = swap_endianness(((uint64_t)origin.a << 32) | (uint64_t)origin.b);
    block[5] = swap_endianness(((uint64_t)origin.c << 32) | (uint64_t)origin.d);
    block[10] = swap_endianness(((uint64_t)origin.e << 32) | (uint64_t)salt.a);
    block[15] = swap_endianness(((uint64_t)salt.b << 32) | (uint64_t)salt.c);
    block[20] = swap_endianness(((uint64_t)salt.d << 32) | (uint64_t)salt.e);
    block[1] = swap_endianness(((uint64_t)salt.f << 32) | (uint64_t)salt.g);
    block[6] = swap_endianness(((uint64_t)salt.h << 32) | (1ULL << 24));

    block[8] = 0x8000000000000000;

    block_permute(block);

    uint64_t a = swap_endianness(block[0]);
    uint64_t b = swap_endianness(block[5]);
    uint64_t c = swap_endianness(block[10]);
    uint64_t d = swap_endianness(block[15]);

    return {(uint32_t)(a >> 32), (uint32_t)(a & 0xFFFFFFFF), (uint32_t)(b >> 32), (uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}