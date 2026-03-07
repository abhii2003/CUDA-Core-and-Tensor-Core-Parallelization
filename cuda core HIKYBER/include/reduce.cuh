#ifndef REDUCE_CUH
#define REDUCE_CUH

#include "params.h"

// Barrett reduction: a -> [0, q]
// Input:  16-bit signed integer a in [-2^15*q, 2^15*q)
// Output: 16-bit signed integer in [0, q]
__device__ __forceinline__
    int16_t
    barrett_reduce(int16_t a)
{
    int16_t t;
    t = (int16_t)(((int32_t)BARRETT_V * a + (1 << 25)) >> BARRETT_SHIFT);
    t *= KYBER_Q;
    return a - t;
}

// Montgomery reduction: a -> (-q, q)
// Input:  32-bit signed integer a in [-2^15*q, 2^15*q)
// Output: 16-bit signed integer in (-q, q)
__device__ __forceinline__
    int16_t
    montgomery_reduce(int32_t a)
{
    int16_t t;
    t = (int16_t)((int16_t)a * (int16_t)QINV);
    t = (int16_t)((a - (int32_t)t * KYBER_Q) >> 16);
    return t;
}

// Fused Montgomery multiplication: (a * b) * R^{-1} mod q
__device__ __forceinline__
    int16_t
    mont_mul(int16_t a, int16_t b)
{
    return montgomery_reduce((int32_t)a * b);
}

// Conditional subtraction of q
__device__ __forceinline__
    int16_t
    csubq(int16_t a)
{
    a -= KYBER_Q;
    a += (a >> 15) & KYBER_Q;
    return a;
}

#ifdef USE_PLANTARD

// Improved Plantard multiplication
// Input:  16-bit signed f, g in [-2^3*q, 2^3*q]
// Output: 16-bit signed h in (-q/2, q/2)
// h = f*g*(-2^{-2l}) mod +- q
__device__ __forceinline__
    int16_t
    plantard_mul(int16_t f, int16_t g)
{
    int32_t fg = (int32_t)f * (int32_t)g;
    int32_t t = (int16_t)((int16_t)(fg) * (int16_t)(PLANTARD_QPRIME & 0xFFFF));
    t = (fg - (int32_t)t * KYBER_Q) >> PLANTARD_L;
    // Extra shift by 2*alpha not needed when twiddle factor is pre-twisted
    return (int16_t)t;
}

// Improved Plantard multiplication with pre-twisted constant
// The constant already incorporates the (-2^{-2l}) factor
__device__ __forceinline__
    int16_t
    plantard_mul_const(int16_t f, int32_t g_twisted)
{
    int32_t fg = (int32_t)f * (int16_t)(g_twisted & 0xFFFF);
    int32_t hi = (int32_t)f * (int16_t)(g_twisted >> 16);
    int32_t t = (int16_t)((int16_t)(fg) * (int16_t)(PLANTARD_QPRIME & 0xFFFF));
    t = (fg - (int32_t)t * KYBER_Q) >> PLANTARD_L;
    t += hi;
    return (int16_t)t;
}

// Improved Plantard reduction
// Input:  32-bit signed f in [-2^6 * q^2, 2^6 * q^2]
// Output: 16-bit signed h in (-q/2, q/2)
__device__ __forceinline__
    int16_t
    plantard_reduce(int32_t f)
{
    int32_t t = (int16_t)((int16_t)(f) * (int16_t)(PLANTARD_QPRIME & 0xFFFF));
    t = (f - (int32_t)t * KYBER_Q) >> PLANTARD_L;
    return (int16_t)t;
}

#endif // USE_PLANTARD

// Reduction wrappers — select suite at compile time
__device__ __forceinline__
    int16_t
    fqmul(int16_t a, int16_t b)
{
#ifdef USE_PLANTARD
    return plantard_mul(a, b);
#else
    return montgomery_reduce((int32_t)a * b);
#endif
}

__device__ __forceinline__
    int16_t
    reduce_short(int16_t a)
{
#ifdef USE_PLANTARD
    return plantard_reduce((int32_t)a);
#else
    return barrett_reduce(a);
#endif
}

#endif // REDUCE_CUH
