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


#define ADD_CC(result, carry_out, x, y) result = x + y; carry_out = (result < x);
#define ADDC_CC(result, carry_out, carry_in, x, y) result = x + y + carry_in; carry_out = (result <= x) ? (result == x ? carry_in : 1) : 0;
#define ADDC(result, carry_in, x, y) result = x + y + carry_in;

#define SUB_CC(result, borrow_out, x, y) result = x - y; borrow_out = (result > x);
#define SUBC_CC(result, borrow_out, borrow_in, x, y) result = x - y - borrow_in; borrow_out = (result >= x) ? (result == x ? borrow_in : 1) : 0;
#define SUBC(result, borrow_in, x, y) result = x - y - borrow_in;

#define MUL_LO(x, y) ((uint32_t)(x * y))
#define MUL_HI(x, y) ((uint32_t)(((uint64_t)x * (uint64_t)y) >> 32))


_uint256 cpu_add_256(_uint256 x, _uint256 y) {
    _uint256 result;

    bool carry;
    ADD_CC(result.h, carry, x.h, y.h);
    ADDC_CC(result.g, carry, carry, x.g, y.g);
    ADDC_CC(result.f, carry, carry, x.f, y.f);
    ADDC_CC(result.e, carry, carry, x.e, y.e);
    ADDC_CC(result.d, carry, carry, x.d, y.d);
    ADDC_CC(result.c, carry, carry, x.c, y.c);
    ADDC_CC(result.b, carry, carry, x.b, y.b);
    ADDC(result.a, carry, x.a, y.a);

    return result;
}


_uint256c cpu_add_256_with_c(_uint256 x, _uint256 y) {
    _uint256c result;

    bool carry;
    ADD_CC(result.h, carry, x.h, y.h);
    ADDC_CC(result.g, carry, carry, x.g, y.g);
    ADDC_CC(result.f, carry, carry, x.f, y.f);
    ADDC_CC(result.e, carry, carry, x.e, y.e);
    ADDC_CC(result.d, carry, carry, x.d, y.d);
    ADDC_CC(result.c, carry, carry, x.c, y.c);
    ADDC_CC(result.b, carry, carry, x.b, y.b);
    ADDC_CC(result.a, carry, carry, x.a, y.a);
    result.carry = carry;

    return result;
}


_uint288c cpu_add_288_with_c(_uint288 x, _uint288 y) {
    _uint288c result;

    bool carry;
    ADD_CC(result.i, carry, x.i, y.i);
    ADDC_CC(result.h, carry, carry, x.h, y.h);
    ADDC_CC(result.g, carry, carry, x.g, y.g);
    ADDC_CC(result.f, carry, carry, x.f, y.f);
    ADDC_CC(result.e, carry, carry, x.e, y.e);
    ADDC_CC(result.d, carry, carry, x.d, y.d);
    ADDC_CC(result.c, carry, carry, x.c, y.c);
    ADDC_CC(result.b, carry, carry, x.b, y.b);
    ADDC_CC(result.a, result.carry, carry, x.a, y.a);

    return result;
}


_uint288c cpu_add_288c_288(_uint288c x, _uint288 y) {
    _uint288c result;

    bool carry;
    ADD_CC(result.i, carry, x.i, y.i);
    ADDC_CC(result.h, carry, carry, x.h, y.h);
    ADDC_CC(result.g, carry, carry, x.g, y.g);
    ADDC_CC(result.f, carry, carry, x.f, y.f);
    ADDC_CC(result.e, carry, carry, x.e, y.e);
    ADDC_CC(result.d, carry, carry, x.d, y.d);
    ADDC_CC(result.c, carry, carry, x.c, y.c);
    ADDC_CC(result.b, carry, carry, x.b, y.b);
    ADDC_CC(result.a, carry, carry, x.a, y.a);
    ADDC(result.carry, carry, x.carry, 0);

    return result;
}


_uint256 cpu_sub_256(_uint256 x, _uint256 y) {
    _uint256 result;

    bool borrow;
    SUB_CC(result.h, borrow, x.h, y.h);
    SUBC_CC(result.g, borrow, borrow, x.g, y.g);
    SUBC_CC(result.f, borrow, borrow, x.f, y.f);
    SUBC_CC(result.e, borrow, borrow, x.e, y.e);
    SUBC_CC(result.d, borrow, borrow, x.d, y.d);
    SUBC_CC(result.c, borrow, borrow, x.c, y.c);
    SUBC_CC(result.b, borrow, borrow, x.b, y.b);
    SUBC(result.a, borrow, x.a, y.a);

    return result;
}


_uint256 cpu_sub_256_mod_p(_uint256 x, _uint256 y) {
    _uint256 result;

    bool borrow;
    SUB_CC(result.h, borrow, x.h, y.h);
    SUBC_CC(result.g, borrow, borrow, x.g, y.g);
    SUBC_CC(result.f, borrow, borrow, x.f, y.f);
    SUBC_CC(result.e, borrow, borrow, x.e, y.e);
    SUBC_CC(result.d, borrow, borrow, x.d, y.d);
    SUBC_CC(result.c, borrow, borrow, x.c, y.c);
    SUBC_CC(result.b, borrow, borrow, x.b, y.b);
    SUBC_CC(result.a, borrow, borrow, x.a, y.a);

    if (borrow != 0) {
        _uint256 result2;

        bool carry = 0;
        ADD_CC(result2.h, carry, result.h, 0xFFFFFC2F);
        ADDC_CC(result2.g, carry, carry, result.g, 0xFFFFFFFE);
        ADDC_CC(result2.f, carry, carry, result.f, 0xFFFFFFFF);
        ADDC_CC(result2.e, carry, carry, result.e, 0xFFFFFFFF);
        ADDC_CC(result2.d, carry, carry, result.d, 0xFFFFFFFF);
        ADDC_CC(result2.c, carry, carry, result.c, 0xFFFFFFFF);
        ADDC_CC(result2.b, carry, carry, result.b, 0xFFFFFFFF);
        ADDC(result2.a, carry, result.a, 0xFFFFFFFF);

        return result2;
    }

    return result;
}


_uint288c cpu_sub_288c_with_c(_uint288c x, _uint288c y) {
    _uint288c result;

    bool borrow;
    SUB_CC(result.i, borrow, x.i, y.i);
    SUBC_CC(result.h, borrow, borrow, x.h, y.h);
    SUBC_CC(result.g, borrow, borrow, x.g, y.g);
    SUBC_CC(result.f, borrow, borrow, x.f, y.f);
    SUBC_CC(result.e, borrow, borrow, x.e, y.e);
    SUBC_CC(result.d, borrow, borrow, x.d, y.d);
    SUBC_CC(result.c, borrow, borrow, x.c, y.c);
    SUBC_CC(result.b, borrow, borrow, x.b, y.b);
    SUBC_CC(result.a, result.carry, borrow, x.a, y.a);

    return result;
}


_uint288 cpu_mul_256_with_word_with_overflow(_uint256 x, uint32_t y) {
    _uint288 result;

    bool carry;
    uint32_t t1;
    uint32_t t2;

    result.i = MUL_LO(x.h, y);
    t1 = MUL_HI(x.h, y);
    t2 = MUL_LO(x.g, y);
    ADD_CC(result.h, carry, t1, t2);
    t1 = MUL_HI(x.g, y);
    t2 = MUL_LO(x.f, y);
    ADDC_CC(result.g, carry, carry, t1, t2);
    t1 = MUL_HI(x.f, y);
    t2 = MUL_LO(x.e, y);
    ADDC_CC(result.f, carry, carry, t1, t2);
    t1 = MUL_HI(x.e, y);
    t2 = MUL_LO(x.d, y);
    ADDC_CC(result.e, carry, carry, t1, t2);
    t1 = MUL_HI(x.d, y);
    t2 = MUL_LO(x.c, y);
    ADDC_CC(result.d, carry, carry, t1, t2);
    t1 = MUL_HI(x.c, y);
    t2 = MUL_LO(x.b, y);
    ADDC_CC(result.c, carry, carry, t1, t2);
    t1 = MUL_HI(x.b, y);
    t2 = MUL_LO(x.a, y);
    ADDC_CC(result.b, carry, carry, t1, t2);
    t1 = MUL_HI(x.a, y);
    ADDC(result.a, carry, t1, 0);

    return result;
}


_uint288c cpu_mul_256_with_word_plus_carry_with_overflow_with_c(_uint256 x, uint32_t y, bool carry) {
    _uint288 result = cpu_mul_256_with_word_with_overflow(x, y);

    if (carry) {
        return cpu_add_288_with_c(result, _uint288{x.a, x.b, x.c, x.d, x.e, x.f, x.g, x.h, 0});
    } else {
        return _uint288c{0, result.a, result.b, result.c, result.d, result.e, result.f, result.g, result.h, result.i};
    }
}


_uint256 cpu_mul_256_mod_p(_uint256 x, _uint256 y) {
    _uint288c z{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    _uint288 t1;
    _uint288c t2;

    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.a);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);

    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.b);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);

    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.c);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);
    
    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.d);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);
    
    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.e);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);
    
    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.f);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);

    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.g);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);
    
    z.carry = (bool)z.a; z.a = z.b; z.b = z.c; z.c = z.d; z.d = z.e; z.e = z.f; z.f = z.g; z.g = z.h; z.h = z.i; z.i = 0;
    t1 = cpu_mul_256_with_word_with_overflow(x, y.h);
    z = cpu_add_288c_288(z, t1);
    t2 = cpu_mul_256_with_word_plus_carry_with_overflow_with_c(P, z.a, z.carry);
    z = cpu_sub_288c_with_c(z, t2);

    if (z.carry || z.a || gte_256(uint288c_to_uint256(z), P)) {
        z = cpu_sub_288c_with_c(z, uint256_to_uint288c(P));
    }

    return _uint256{z.b, z.c, z.d, z.e, z.f, z.g, z.h, z.i};
}


_uint256 cpu_rshift1_256(_uint256 x) {
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

_uint256 cpu_rshift1_256c(_uint256c x) {
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


_uint256 cpu_eeuclid_256_mod_p(_uint256 input) {
    // https://www.researchgate.net/publication/344657706_A_New_Improvement_of_Extended_Stein's_Binary_Algorithm
    _uint256 u = input;
    _uint256 v = P;
    _uint256 x{0, 0, 0, 0, 0, 0, 0, 1};
    _uint256 y{0, 0, 0, 0, 0, 0, 0, 0};

    while ((u.h & 1) == 0) {
        u = cpu_rshift1_256(u);

        _uint256c x_;
        if ((x.h & 1) == 1) {
            x_ = cpu_add_256_with_c(x, P);
        } else {
            x_ = uint256_to_uint256c(x);
        }
        x = cpu_rshift1_256c(x_);
    }

    while (neq_256(u, v)) {
        if (gt_256(u, v)) {
            u = cpu_sub_256(u, v);
            x = cpu_sub_256_mod_p(x, y);

            while ((u.h & 1) == 0) {
                u = cpu_rshift1_256(u);

                _uint256c x_;
                if ((x.h & 1) == 1) {
                    x_ = cpu_add_256_with_c(x, P);
                } else {
                    x_ = uint256_to_uint256c(x);
                }
                x = cpu_rshift1_256c(x_);
            }
        } else {
            v = cpu_sub_256(v, u);
            y = cpu_sub_256_mod_p(y, x);

            while ((v.h & 1) == 0) {
                v = cpu_rshift1_256(v);

                _uint256c y_;
                if ((y.h & 1) == 1) {
                    y_ = cpu_add_256_with_c(y, P);
                } else {
                    y_ = uint256_to_uint256c(y);
                }
                y = cpu_rshift1_256c(y_);
            }
        }
    }

    return x;
}