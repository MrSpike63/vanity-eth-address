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
#if defined(_WIN64)
    #include <bcrypt.h>
    #include <ntstatus.h>
#endif
#include <cinttypes>
#include <iostream>

#include "structures.h"


int generate_secure_random_key(_uint256& key, _uint256 max, int num_bits) {
    int num_bytes = (num_bits + 7) / 8;
    int full_bytes = num_bits >> 3;
    uint8_t* bytes_buffer = new uint8_t[num_bytes];

    #if defined(_WIN64)
        while (true) {
            NTSTATUS status = BCryptGenRandom(0, bytes_buffer, num_bytes, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            if (status == STATUS_SUCCESS) {
                #define fetch(n) ((uint32_t)((n < full_bytes) ? bytes_buffer[n] : ((n == full_bytes && num_bytes != full_bytes) ? (bytes_buffer[n] >> (8 - (num_bits & 7))) : 0)))
                key.h = (fetch(3) << 24) | (fetch(2) << 16) | (fetch(1) << 8) | (fetch(0));
                key.g = (fetch(7) << 24) | (fetch(6) << 16) | (fetch(5) << 8) | (fetch(4));
                key.f = (fetch(11) << 24) | (fetch(10) << 16) | (fetch(9) << 8) | (fetch(8));
                key.e = (fetch(15) << 24) | (fetch(14) << 16) | (fetch(13) << 8) | (fetch(12));
                key.d = (fetch(19) << 24) | (fetch(18) << 16) | (fetch(17) << 8) | (fetch(16));
                key.c = (fetch(23) << 24) | (fetch(22) << 16) | (fetch(21) << 8) | (fetch(20));
                key.b = (fetch(27) << 24) | (fetch(26) << 16) | (fetch(25) << 8) | (fetch(24));
                key.a = (fetch(31) << 24) | (fetch(30) << 16) | (fetch(29) << 8) | (fetch(28));
                #undef fetch

                if (!gt_256(key, max)) {
                    delete[] bytes_buffer;
                    return 0;
                }
            } else {
                delete[] bytes_buffer;
                return 1;
            }
        }
    #elif defined(__linux__)
        FILE* fp = fopen("/dev/urandom", "rb");
        if (fp) {
            while (true) {
                int read = fread(bytes_buffer, 1, num_bytes, fp);

                if (read == num_bytes) {
                    #define fetch(n) ((n < full_bytes) ? bytes_buffer[n] : ((n == full_bytes && num_bytes != full_bytes) ? (bytes_buffer[n] >> (8 - (num_bits & 7))) : 0))
                    key.h = (fetch(3) << 24) | (fetch(2) << 16) | (fetch(1) << 8) | (fetch(0));
                    key.g = (fetch(7) << 24) | (fetch(6) << 16) | (fetch(5) << 8) | (fetch(4));
                    key.f = (fetch(11) << 24) | (fetch(10) << 16) | (fetch(9) << 8) | (fetch(8));
                    key.e = (fetch(15) << 24) | (fetch(14) << 16) | (fetch(13) << 8) | (fetch(12));
                    key.d = (fetch(19) << 24) | (fetch(18) << 16) | (fetch(17) << 8) | (fetch(16));
                    key.c = (fetch(23) << 24) | (fetch(22) << 16) | (fetch(21) << 8) | (fetch(20));
                    key.b = (fetch(27) << 24) | (fetch(26) << 16) | (fetch(25) << 8) | (fetch(24));
                    key.a = (fetch(31) << 24) | (fetch(30) << 16) | (fetch(29) << 8) | (fetch(28));
                    #undef fetch

                    if (!gt_256(key, max)) {
                        fclose(fp);
                        delete[] bytes_buffer;
                        return 0;
                    }
                } else {
                    fclose(fp);
                    delete[] bytes_buffer;
                    return 2;
                }
            }
        } else {
            return 3;
        }
    #else
        #error No secure random implementation for the target platform
    #endif
}