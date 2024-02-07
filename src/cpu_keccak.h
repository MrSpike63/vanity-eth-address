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


uint64_t cpu_rotate(uint64_t x, int n) {
    return (x << n) | (x >> (64 - n));
}

uint64_t cpu_swap_endianness(uint64_t x) {
    return ((x & 0x00000000000000FF) << 56) | ((x & 0x000000000000FF00) << 40) | ((x & 0x0000000000FF0000) << 24) | ((x & 0x00000000FF000000) << 8) | ((x & 0x000000FF00000000) >> 8) | ((x & 0x0000FF0000000000) >> 24) | ((x & 0x00FF000000000000) >> 40) | ((x & 0xFF00000000000000) >> 56);
}

const uint64_t CPU_IOTA_CONSTANTS[24] = {
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003, 0x8000000000008002, 0x8000000000000080,0x000000000000800A, 0x800000008000000A, 0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
};


void cpu_block_permute(uint64_t *block) {
    uint64_t C[5];
    uint64_t D;
    uint64_t temp1, temp2;

    for (int t = 0; t < 24; t++) {
        C[0] = block[0] ^ block[1] ^ block[2] ^ block[3] ^ block[4];
        C[1] = block[5] ^ block[6] ^ block[7] ^ block[8] ^ block[9];
        C[2] = block[10] ^ block[11] ^ block[12] ^ block[13] ^ block[14];
        C[3] = block[15] ^ block[16] ^ block[17] ^ block[18] ^ block[19];
        C[4] = block[20] ^ block[21] ^ block[22] ^ block[23] ^ block[24];

        D = C[4] ^ cpu_rotate(C[1], 1);
        block[0] ^= D; block[1] ^= D; block[2] ^= D; block[3] ^= D; block[4] ^= D;
        D = C[0] ^ cpu_rotate(C[2], 1);
        block[5] ^= D; block[6] ^= D; block[7] ^= D; block[8] ^= D; block[9] ^= D;
        D = C[1] ^ cpu_rotate(C[3], 1);
        block[10] ^= D; block[11] ^= D; block[12] ^= D; block[13] ^= D; block[14] ^= D;
        D = C[2] ^ cpu_rotate(C[4], 1);
        block[15] ^= D; block[16] ^= D; block[17] ^= D; block[18] ^= D; block[19] ^= D;
        D = C[3] ^ cpu_rotate(C[0], 1);
        block[20] ^= D; block[21] ^= D; block[22] ^= D; block[23] ^= D; block[24] ^= D;


        temp1 = block[8];
        block[8] = cpu_rotate(block[1], 36);  
        block[1] = cpu_rotate(block[15], 28); 
        block[15] = cpu_rotate(block[18], 21);
        block[18] = cpu_rotate(block[13], 15);
        block[13] = cpu_rotate(block[7], 10); 
        block[7] = cpu_rotate(block[11], 6);  
        block[11] = cpu_rotate(block[2], 3);  
        block[2] = cpu_rotate(block[5], 1);   
        block[5] = cpu_rotate(block[6], 44);  
        block[6] = cpu_rotate(block[21], 20); 
        block[21] = cpu_rotate(block[14], 61);
        block[14] = cpu_rotate(block[22], 39);
        block[22] = cpu_rotate(block[4], 18);
        block[4] = cpu_rotate(block[10], 62);
        block[10] = cpu_rotate(block[12], 43);
        block[12] = cpu_rotate(block[17], 25);
        block[17] = cpu_rotate(block[23], 8);
        block[23] = cpu_rotate(block[19], 56);
        block[19] = cpu_rotate(block[3], 41);
        block[3] = cpu_rotate(block[20], 27);
        block[20] = cpu_rotate(block[24], 14);
        block[24] = cpu_rotate(block[9], 2);
        block[9] = cpu_rotate(block[16], 55);
        block[16] = cpu_rotate(temp1, 45);


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


        block[0] ^= CPU_IOTA_CONSTANTS[t];
    }
}


Address cpu_calculate_address(_uint256 x, _uint256 y) {
    uint64_t block[50];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = cpu_swap_endianness(((uint64_t)x.a << 32) | x.b);
    block[5] = cpu_swap_endianness(((uint64_t)x.c << 32) | x.d);
    block[10] = cpu_swap_endianness(((uint64_t)x.e << 32) | x.f);
    block[15] = cpu_swap_endianness(((uint64_t)x.g << 32) | x.h);
    block[20] = cpu_swap_endianness(((uint64_t)y.a << 32) | y.b);
    block[1] = cpu_swap_endianness(((uint64_t)y.c << 32) | y.d);
    block[6] = cpu_swap_endianness(((uint64_t)y.e << 32) | y.f);
    block[11] = cpu_swap_endianness(((uint64_t)y.g << 32) | y.h);
    block[16] = (1ULL << 0);

    block[8] = 0x8000000000000000;

    cpu_block_permute(block);

    uint64_t b = cpu_swap_endianness(block[5]);
    uint64_t c = cpu_swap_endianness(block[10]);
    uint64_t d = cpu_swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}


Address cpu_calculate_contract_address(Address a, uint8_t nonce = 0x80) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = cpu_swap_endianness((0xD694ULL << 48) | ((uint64_t)a.a << 16) | (a.b >> 16));
    block[5] = cpu_swap_endianness(((uint64_t)a.b << 48) | ((uint64_t)a.c << 16) | (a.d >> 16));
    block[10] = cpu_swap_endianness(((uint64_t)a.d << 48) | ((uint64_t)a.e << 16) | ((uint64_t)nonce << 8) | 1);

    block[8] = 0x8000000000000000;

    cpu_block_permute(block);

    uint64_t b = cpu_swap_endianness(block[5]);
    uint64_t c = cpu_swap_endianness(block[10]);
    uint64_t d = cpu_swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}


_uint256 cpu_full_keccak(uint8_t* bytes, uint32_t num_bytes) {
    int input_blocks = (num_bytes + 136 - 1 + 1) / 136;

    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }


    #define fetch(n) ((i * 136 + n < num_bytes) ? bytes[i * 136 + n] : ((i * 136 + n == num_bytes) ? 1 : 0))
    #define block_xor(block_num, n) block[block_num] ^= cpu_swap_endianness(((uint64_t)fetch(n * 8 + 0) << 56) | ((uint64_t)fetch(n * 8 + 1) << 48) | ((uint64_t)fetch(n * 8 + 2) << 40) | ((uint64_t)fetch(n * 8 + 3) << 32) | ((uint64_t)fetch(n * 8 + 4) << 24) | ((uint64_t)fetch(n * 8 + 5) << 16) | ((uint64_t)fetch(n * 8 + 6) << 8) | ((uint64_t)fetch(n * 8 + 7)))
    for (int i = 0; i < input_blocks; i++) {
        block_xor(0, 0);
        block_xor(5, 1);
        block_xor(10, 2);
        block_xor(15, 3);
        block_xor(20, 4);
        block_xor(1, 5);
        block_xor(6, 6);
        block_xor(11, 7);
        block_xor(16, 8);
        block_xor(21, 9);
        block_xor(2, 10);
        block_xor(7, 11);
        block_xor(12, 12);
        block_xor(17, 13);
        block_xor(22, 14);
        block_xor(3, 15);
        block_xor(8, 16);

        if (i == input_blocks - 1) {
            block[8] ^= 0x8000000000000000;
        }

        cpu_block_permute(block);
    }
    #undef fetch

    uint64_t a = cpu_swap_endianness(block[0]);
    uint64_t b = cpu_swap_endianness(block[5]);
    uint64_t c = cpu_swap_endianness(block[10]);
    uint64_t d = cpu_swap_endianness(block[15]);

    return {(uint32_t)(a >> 32), (uint32_t)(a & 0xFFFFFFFF), (uint32_t)(b >> 32), (uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}


Address cpu_calculate_contract_address2(Address a, _uint256 salt, _uint256 bytecode) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = cpu_swap_endianness((0xFFULL << 56) | ((uint64_t)a.a << 24) | (a.b >> 8));
    block[5] = cpu_swap_endianness(((uint64_t)a.b << 56) | ((uint64_t)a.c << 24) | (a.d >> 8));
    block[10] = cpu_swap_endianness(((uint64_t)a.d << 56) | ((uint64_t)a.e << 24) | (salt.a >> 8));
    block[15] = cpu_swap_endianness(((uint64_t)salt.a << 56) | ((uint64_t)salt.b << 24) | (salt.c >> 8));
    block[20] = cpu_swap_endianness(((uint64_t)salt.c << 56) | ((uint64_t)salt.d << 24) | (salt.e >> 8));
    block[1] = cpu_swap_endianness(((uint64_t)salt.e << 56) | ((uint64_t)salt.f << 24) | (salt.g >> 8));
    block[6] = cpu_swap_endianness(((uint64_t)salt.g << 56) | ((uint64_t)salt.h << 24) | (bytecode.a >> 8));
    block[11] = cpu_swap_endianness(((uint64_t)bytecode.a << 56) | ((uint64_t)bytecode.b << 24) | (bytecode.c >> 8));
    block[16] = cpu_swap_endianness(((uint64_t)bytecode.c << 56) | ((uint64_t)bytecode.d << 24) | (bytecode.e >> 8));
    block[21] = cpu_swap_endianness(((uint64_t)bytecode.e << 56) | ((uint64_t)bytecode.f << 24) | (bytecode.g >> 8));
    block[2] = cpu_swap_endianness(((uint64_t)bytecode.g << 56) | ((uint64_t)bytecode.h << 24) | (1 << 16));

    block[8] = 0x8000000000000000;

    cpu_block_permute(block);

    uint64_t b = cpu_swap_endianness(block[5]);
    uint64_t c = cpu_swap_endianness(block[10]);
    uint64_t d = cpu_swap_endianness(block[15]);

    return {(uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}

_uint256 cpu_calculate_create3_salt(Address origin, _uint256 salt) {
    uint64_t block[25];
    for (int i = 0; i < 25; i++) {
        block[i] = 0;
    }

    block[0] = cpu_swap_endianness(((uint64_t)origin.a << 32) | (uint64_t)origin.b);
    block[5] = cpu_swap_endianness(((uint64_t)origin.c << 32) | (uint64_t)origin.d);
    block[10] = cpu_swap_endianness(((uint64_t)origin.e << 32) | (uint64_t)salt.a);
    block[15] = cpu_swap_endianness(((uint64_t)salt.b << 32) | (uint64_t)salt.c);
    block[20] = cpu_swap_endianness(((uint64_t)salt.d << 32) | (uint64_t)salt.e);
    block[1] = cpu_swap_endianness(((uint64_t)salt.f << 32) | (uint64_t)salt.g);
    block[6] = cpu_swap_endianness(((uint64_t)salt.h << 32) | (1ULL << 24));

    block[8] = 0x8000000000000000;

    cpu_block_permute(block);

    uint64_t a = cpu_swap_endianness(block[0]);
    uint64_t b = cpu_swap_endianness(block[5]);
    uint64_t c = cpu_swap_endianness(block[10]);
    uint64_t d = cpu_swap_endianness(block[15]);

    return {(uint32_t)(a >> 32), (uint32_t)(a & 0xFFFFFFFF), (uint32_t)(b >> 32), (uint32_t)(b & 0xFFFFFFFF), (uint32_t)(c >> 32), (uint32_t)(c & 0xFFFFFFFF), (uint32_t)(d >> 32), (uint32_t)(d & 0xFFFFFFFF)};
}