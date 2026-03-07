#ifndef POLY_CUH
#define POLY_CUH

#include "params.h"
#include "reduce.cuh"
#include "ntt.cuh"

struct poly
{
    int16_t coeffs[KYBER_N];
};

struct polyvec
{
    poly vec[KYBER_K];
};

// NTT forward transform (EDFS)
__device__ __forceinline__ void poly_ntt(poly *p)
{
    ntt_edfs(p->coeffs);
}

// INTT (SLM)
__device__ __forceinline__ void poly_intt(poly *p)
{
    intt_slm(p->coeffs);
}

// ---- Pointwise multiplication ----
// Method I (Karatsuba) from paper Section III-E
// Computes basemul of degree-1 polynomials in Zq[X]/(X^2 - zeta)
// Input: f0+f1*X, g0+g1*X   Output: h0+h1*X
// Uses 4 multiplications + 4 reductions
__device__ __forceinline__ void basemul(int16_t *h, const int16_t *f, const int16_t *g, int16_t zeta)
{
    // a = f0*g0, b = f1*g1 (reusable products, Karatsuba)
    int32_t a = (int32_t)f[0] * g[0];
    int32_t b = (int32_t)f[1] * g[1];

    // Memory access optimization from Fig. 9: compute additions early
    int16_t c = f[0] + f[1]; // f0 + f1
    int16_t d = g[0] + g[1]; // g0 + g1

    // h0 = a + b*zeta  (reduced)
    h[0] = montgomery_reduce(a + montgomery_reduce(b) * (int32_t)zeta);
    // h1 = c*d - a - b  = (f0+f1)(g0+g1) - f0*g0 - f1*g1
    int32_t cd = (int32_t)c * d;
    h[1] = montgomery_reduce(cd - a - b);
}

// Pointwise multiplication of two polynomials in NTT domain
__device__ void poly_basemul(poly *r, const poly *a, const poly *b)
{
    for (int i = 0; i < KYBER_N / 4; i++)
    {
        basemul(&r->coeffs[4 * i],
                &a->coeffs[4 * i],
                &b->coeffs[4 * i],
                h_zetas[64 + i]); // zeta^{2*br7(i)+1}
        basemul(&r->coeffs[4 * i + 2],
                &a->coeffs[4 * i + 2],
                &b->coeffs[4 * i + 2],
                -h_zetas[64 + i]); // negated for second pair
    }
}

// Multiply-accumulate: r += a[i] * b[i] for polyvec
__device__ void polyvec_basemul_acc(poly *r, const polyvec *a, const polyvec *b)
{
    poly tmp;
    poly_basemul(r, &a->vec[0], &b->vec[0]);
    for (int i = 1; i < KYBER_K; i++)
    {
        poly_basemul(&tmp, &a->vec[i], &b->vec[i]);
        for (int j = 0; j < KYBER_N; j++)
            r->coeffs[j] += tmp.coeffs[j];
    }
    // Reduce after accumulation
    for (int j = 0; j < KYBER_N; j++)
        r->coeffs[j] = barrett_reduce(r->coeffs[j]);
}

// Coefficient-wise addition
__device__ __forceinline__ void poly_add(poly *r, const poly *a, const poly *b)
{
    for (int i = 0; i < KYBER_N; i++)
        r->coeffs[i] = a->coeffs[i] + b->coeffs[i];
}

// Coefficient-wise subtraction
__device__ __forceinline__ void poly_sub(poly *r, const poly *a, const poly *b)
{
    for (int i = 0; i < KYBER_N; i++)
        r->coeffs[i] = a->coeffs[i] - b->coeffs[i];
}

// Reduce all coefficients
__device__ __forceinline__ void poly_reduce(poly *p)
{
    for (int i = 0; i < KYBER_N; i++)
        p->coeffs[i] = barrett_reduce(p->coeffs[i]);
}

// Conditional subtraction
__device__ __forceinline__ void poly_csubq(poly *p)
{
    for (int i = 0; i < KYBER_N; i++)
        p->coeffs[i] = csubq(p->coeffs[i]);
}

// NTT / INTT for polyvec
__device__ __forceinline__ void polyvec_ntt(polyvec *pv)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_ntt(&pv->vec[i]);
}

__device__ __forceinline__ void polyvec_intt(polyvec *pv)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_intt(&pv->vec[i]);
}

__device__ __forceinline__ void polyvec_reduce(polyvec *pv)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_reduce(&pv->vec[i]);
}

__device__ __forceinline__ void polyvec_csubq(polyvec *pv)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_csubq(&pv->vec[i]);
}

__device__ __forceinline__ void polyvec_add(polyvec *r, const polyvec *a, const polyvec *b)
{
    for (int i = 0; i < KYBER_K; i++)
        poly_add(&r->vec[i], &a->vec[i], &b->vec[i]);
}

#endif // POLY_CUH
